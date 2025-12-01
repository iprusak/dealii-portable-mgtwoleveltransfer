#ifndef portable_polynomial_transfer_h
#define portable_polynomial_transfer_h

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/matrix_free/constraint_info.h>
#include <deal.II/matrix_free/shape_info.h>

#include <Kokkos_Core.hpp>

#include "portable_polynomial_transfer_base.h"

DEAL_II_NAMESPACE_OPEN

namespace Portable {

template <int dim, typename number> struct TransferData {
  const typename MatrixFree<dim, number>::PrecomputedData gpu_data_coarse;
  const typename MatrixFree<dim, number>::PrecomputedData gpu_data_fine;
  const Kokkos::View<number **, MemorySpace::Default::kokkos_space> &weights;
  const Kokkos::View<number *, MemorySpace::Default::kokkos_space>
      &prolongation_matrix;
  const Kokkos::View<int *, MemorySpace::Default::kokkos_space>
      &cell_lists_fine_to_coarse;
  const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
      &boundary_dofs_mask_coarse;
  const Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>
      &boundary_dofs_mask_fine;
};

template <int dim, int p_coarse, int p_fine, typename number>
class CellProlongationKernel : public EnableObserverPointer {
public:
  CellProlongationKernel(
      TransferData<dim, number> transfer_data,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
          &src,
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst);

  using TeamHandle = Kokkos::TeamPolicy<
      MemorySpace::Default::kokkos_space::execution_space>::member_type;
  using SharedView = Kokkos::View<
      number *,
      MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  std::size_t team_shmem_size(int team_size) const;

  DEAL_II_HOST_DEVICE
  void operator()(const TeamHandle &team_member) const;

  static const unsigned int n_local_dofs_coarse =
      Utilities::pow(p_coarse + 1, dim);
  static const unsigned int n_local_dofs_fine = Utilities::pow(p_fine + 1, dim);

private:
  TransferData<dim, number> transfer_data;

  const DeviceVector<number> src;
  DeviceVector<number> dst;
};

template <int dim, int p_coarse, int p_fine, typename number>
CellProlongationKernel<dim, p_coarse, p_fine, number>::CellProlongationKernel(
    TransferData<dim, number> transfer_data,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst)
    : transfer_data(transfer_data),
      src(src.get_values(), src.locally_owned_size()),
      dst(dst.get_values(), dst.locally_owned_size()) {}

template <int dim, int p_coarse, int p_fine, typename number>
std::size_t
CellProlongationKernel<dim, p_coarse, p_fine, number>::team_shmem_size(
    int /*team_size*/) const {
  return SharedView::shmem_size(
      n_local_dofs_coarse + // coarse dof values
      n_local_dofs_fine +   // fine dof values
      2 * n_local_dofs_fine // at most two tmp vectors of at most
                            // n_local_dofs_fine size
  );
}

DEAL_II_HOST_DEVICE
template <int dim, int p_coarse, int p_fine, typename number>
void CellProlongationKernel<dim, p_coarse, p_fine, number>::operator()(
    const TeamHandle &team_member) const {
  const int cell_index_fine = team_member.league_rank();
  const int cell_index_coarse =
      transfer_data.cell_lists_fine_to_coarse[cell_index_fine];

  SharedView values_coarse(team_member.team_shmem(), n_local_dofs_coarse);
  SharedView values_fine(team_member.team_shmem(), n_local_dofs_fine);

  // read coarse dof values
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(team_member, n_local_dofs_coarse),
      [&](const int &i) {
        values_coarse(i) = src[transfer_data.gpu_data_coarse.local_to_global(
            i, cell_index_coarse)];
      });
  team_member.team_barrier();

  // interpolation tensor-product prolongation kernel
  internal::EvaluatorTensorProduct<internal::EvaluatorVariant::evaluate_general,
                                   dim, p_coarse + 1, p_fine + 1, number>
      prolongation_kernel(
          team_member,
          /*shape_values=*/transfer_data.prolongation_matrix,
          /*shape_gradients=*/
          Kokkos::View<number *, MemorySpace::Default::kokkos_space>(),
          /*co_shape_gradients=*/
          Kokkos::View<number *, MemorySpace::Default::kokkos_space>(),
          SharedView() // the evaluator does not need temporary
                       // storage since no in-place operation takes
                       // place in this function
      );

  // apply kernel in each direction
  if constexpr (dim == 2) {
    auto tmp =
        SharedView(team_member.team_shmem(), (p_coarse + 1) * (p_fine + 1));

    // <direction, dof_to_quad, add, in_place>
    // dof_to_quad == contract_over_rows
    prolongation_kernel.template values<0, true, false, false>(values_coarse,
                                                               tmp);

    prolongation_kernel.template values<1, true, false, false>(tmp,
                                                               values_fine);
  } else if constexpr (dim == 3) {
    auto tmp1 = SharedView(team_member.team_shmem(),
                           Utilities::pow(p_coarse + 1, 2) * (p_fine + 1));

    auto tmp2 = SharedView(team_member.team_shmem(),
                           Utilities::pow(p_fine + 1, 2) * (p_coarse + 1));

    prolongation_kernel.template values<0, true, false, false>(values_coarse,
                                                               tmp1);
    prolongation_kernel.template values<1, true, false, false>(tmp1, tmp2);
    prolongation_kernel.template values<2, true, false, false>(tmp2,
                                                               values_fine);
  }

  // apply weights
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n_local_dofs_fine),
                       [&](const int &i) {
                         values_fine(i) *=
                             transfer_data.weights(i, cell_index_fine);
                       });
  team_member.team_barrier();

  // distribute fine dofs values
  if (transfer_data.gpu_data_fine.use_coloring)
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, n_local_dofs_fine),
        [&](const int &i) {
          if (transfer_data.boundary_dofs_mask_fine(i, cell_index_fine) !=
              numbers::invalid_unsigned_int)
            dst[transfer_data.gpu_data_fine.local_to_global(
                i, cell_index_fine)] += values_fine(i);
        });
  else
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, n_local_dofs_fine),
        [&](const int &i) {
          if (transfer_data.boundary_dofs_mask_fine(i, cell_index_fine) !=
              numbers::invalid_unsigned_int)
            Kokkos::atomic_add(&dst[transfer_data.gpu_data_fine.local_to_global(
                                   i, cell_index_fine)],
                               values_fine(i));
        });
  team_member.team_barrier();
}

template <int dim, int p_coarse, int p_fine, typename number>
class CellRestrictionKernel : public EnableObserverPointer {
public:
  CellRestrictionKernel(
      TransferData<dim, number> transfer_data,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
          &src,
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst);

  using TeamHandle = Kokkos::TeamPolicy<
      MemorySpace::Default::kokkos_space::execution_space>::member_type;
  using SharedView = Kokkos::View<
      number *,
      MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  std::size_t team_shmem_size(int team_size) const;

  DEAL_II_HOST_DEVICE
  void operator()(const TeamHandle &team_member) const;

  static const unsigned int n_local_dofs_coarse =
      Utilities::pow(p_coarse + 1, dim);
  static const unsigned int n_local_dofs_fine = Utilities::pow(p_fine + 1, dim);

private:
  TransferData<dim, number> transfer_data;

  const DeviceVector<number> src;
  DeviceVector<number> dst;
};

template <int dim, int p_coarse, int p_fine, typename number>
CellRestrictionKernel<dim, p_coarse, p_fine, number>::CellRestrictionKernel(
    TransferData<dim, number> transfer_data,
    const LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &src,
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst)
    : transfer_data(transfer_data),
      src(src.get_values(), src.locally_owned_size()),
      dst(dst.get_values(), dst.locally_owned_size()) {}

template <int dim, int p_coarse, int p_fine, typename number>
std::size_t
CellRestrictionKernel<dim, p_coarse, p_fine, number>::team_shmem_size(
    int /*team_size*/) const {
  return SharedView::shmem_size(
      n_local_dofs_coarse + // coarse dof values
      n_local_dofs_fine +   // fine dof values
      2 * n_local_dofs_fine // at most two tmp vectors of at most
                            // n_local_dofs_fine size
  );
}

DEAL_II_HOST_DEVICE
template <int dim, int p_coarse, int p_fine, typename number>
void CellRestrictionKernel<dim, p_coarse, p_fine, number>::operator()(
    const TeamHandle &team_member) const {
  const int cell_index_fine = team_member.league_rank();
  const int cell_index_coarse =
      transfer_data.cell_lists_fine_to_coarse[cell_index_fine];

  SharedView values_fine(team_member.team_shmem(), n_local_dofs_fine);
  SharedView values_coarse(team_member.team_shmem(), n_local_dofs_coarse);

  // read fine dof values
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n_local_dofs_fine),
                       [&](const int &i) {
                         values_fine(i) =
                             src[transfer_data.gpu_data_fine.local_to_global(
                                 i, cell_index_fine)];
                       });
  team_member.team_barrier();

  // apply weights
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team_member, n_local_dofs_fine),
                       [&](const int &i) {
                         values_fine(i) *=
                             transfer_data.weights(i, cell_index_fine);
                       });
  team_member.team_barrier();

  // interpolation tensor-product restriction kernel
  internal::EvaluatorTensorProduct<internal::EvaluatorVariant::evaluate_general,
                                   dim, p_coarse + 1, p_fine + 1, number>
      restriction_kernel(
          team_member,
          /*shape_values=*/transfer_data.prolongation_matrix,
          /*shape_gradients=*/
          Kokkos::View<number *, MemorySpace::Default::kokkos_space>(),
          /*co_shape_gradients=*/
          Kokkos::View<number *, MemorySpace::Default::kokkos_space>(),
          SharedView() // the evaluator does not need temporary
                       // storage since no in-place operation takes
                       // place in this function
      );

  // apply kernel in each direction
  if constexpr (dim == 2) {
    auto tmp =
        SharedView(team_member.team_shmem(), (p_fine + 1) * (p_coarse + 1));

    // <direction, dof_to_quad, add, in_place>
    // dof_to_quad == contract_over_rows
    restriction_kernel.template values<1, false, false, false>(values_fine,
                                                               tmp);

    restriction_kernel.template values<0, false, false, false>(tmp,
                                                               values_coarse);
  } else if constexpr (dim == 3) {
    auto tmp1 = SharedView(team_member.team_shmem(),
                           Utilities::pow(p_fine + 1, 2) * (p_coarse + 1));

    auto tmp2 = SharedView(team_member.team_shmem(),
                           Utilities::pow(p_coarse + 1, 2) * (p_fine + 1));

    restriction_kernel.template values<2, false, false, false>(values_fine,
                                                               tmp1);
    restriction_kernel.template values<1, false, false, false>(tmp1, tmp2);
    restriction_kernel.template values<0, false, false, false>(tmp2,
                                                               values_coarse);
  }

  // distribute coarse dofs values
  if (transfer_data.gpu_data_coarse.use_coloring)
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, n_local_dofs_coarse),
        [&](const int &i) {
          if (transfer_data.boundary_dofs_mask_coarse(i, cell_index_coarse) !=
              numbers::invalid_unsigned_int)
            dst[transfer_data.gpu_data_coarse.local_to_global(
                i, cell_index_coarse)] += values_coarse(i);
        });
  else
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, n_local_dofs_coarse),
        [&](const int &i) {
          if (transfer_data.boundary_dofs_mask_coarse(i, cell_index_coarse) !=
              numbers::invalid_unsigned_int)
            Kokkos::atomic_add(
                &dst[transfer_data.gpu_data_coarse.local_to_global(
                    i, cell_index_coarse)],
                values_coarse(i));
        });
  team_member.team_barrier();
}

template <int dim, int p_coarse, int p_fine, typename number,
          bool overlap_communication_computation>
class PolynomialTransfer
    : public PolynomialTransferBase<dim, number,
                                    overlap_communication_computation> {
public:
  PolynomialTransfer();

  void prolongate_and_add(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
          &src) const override;

  void restrict_and_add(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
      const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
          &src) const override;

  void reinit(const MatrixFree<dim, number> &mf_coarse,
              const MatrixFree<dim, number> &mf_fine,
              const AffineConstraints<number> &constraints_coarse,
              const AffineConstraints<number> &constraints_fine) override;

private:
  void setup_weights_and_boundary_dofs_masks();

  ObserverPointer<const MatrixFree<dim, number>> matrix_free_coarse;
  ObserverPointer<const MatrixFree<dim, number>> matrix_free_fine;

  ObserverPointer<const AffineConstraints<number>> constraints_fine;
  ObserverPointer<const AffineConstraints<number>> constraints_coarse;

  Kokkos::View<number *, MemorySpace::Default::kokkos_space>
      prolongation_matrix_1d;

  std::vector<Kokkos::View<int *, MemorySpace::Default::kokkos_space>>
      cell_lists_fine_to_coarse;

  std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
      boundary_dofs_mask_coarse;

  std::vector<Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>>
      boundary_dofs_mask_fine;

  std::vector<Kokkos::View<number **, MemorySpace::Default::kokkos_space>>
      weights_view_kokkos;
};

template <int dim, int p_coarse, int p_fine, typename number,
          bool overlap_communication_computation>
PolynomialTransfer<dim, p_coarse, p_fine, number,
                   overlap_communication_computation>::PolynomialTransfer() {}

template <int dim, int p_coarse, int p_fine, typename number,
          bool overlap_communication_computation>
void PolynomialTransfer<dim, p_coarse, p_fine, number,
                        overlap_communication_computation>::
    prolongate_and_add(
        LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
        const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
            &src) const {
  Assert(dst.get_partitioner() == matrix_free_fine->get_vector_partitioner(),
         ExcMessage("Fine vector is not initialized correctly."));
  Assert(src.get_partitioner() == matrix_free_coarse->get_vector_partitioner(),
         ExcMessage("Coarse vector is not initialized correctly."));

  MemorySpace::Default::kokkos_space::execution_space exec;

  const auto &colored_graph = matrix_free_fine->get_colored_graph();
  const unsigned int n_colors = colored_graph.size();

  if (overlap_communication_computation) {
    auto do_color = [&](const unsigned int color) {
      const auto &gpu_data_coarse = matrix_free_coarse->get_data(0, color);
      const auto &gpu_data_fine = matrix_free_fine->get_data(0, color);

      const auto n_cells = gpu_data_fine.n_cells;

      Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>
          team_policy(exec, n_cells, Kokkos::AUTO);

      TransferData<dim, number> transfer_data{gpu_data_coarse,
                                              gpu_data_fine,
                                              weights_view_kokkos[color],
                                              prolongation_matrix_1d,
                                              cell_lists_fine_to_coarse[color],
                                              boundary_dofs_mask_coarse[color],
                                              boundary_dofs_mask_fine[color]};

      CellProlongationKernel<dim, p_coarse, p_fine, number> prolongator(
          transfer_data, src, dst);

      Kokkos::parallel_for("prolongate_" + std::to_string(color), team_policy,
                           prolongator);
    };

    src.update_ghost_values_start(0);

    if (n_colors > 0 && colored_graph[0].size() > 0)
      do_color(0);

    src.update_ghost_values_finish();

    if (n_colors > 1 && colored_graph[1].size() > 0) {
      do_color(1);

      // We need a synchronization point because we don't want
      // device-aware MPI to start the MPI communication until the
      // kernel is done.
      Kokkos::fence();
    }
    dst.compress_start(0, VectorOperation::add);

    if (n_colors > 2 && colored_graph[2].size() > 0)
      do_color(2);

    dst.compress_finish(VectorOperation::add);
  } else {
    src.update_ghost_values();

    for (unsigned int color = 0; color < n_colors; ++color) {
      const auto &gpu_data_coarse = matrix_free_coarse->get_data(0, color);
      const auto &gpu_data_fine = matrix_free_fine->get_data(0, color);

      const auto n_cells = gpu_data_fine.n_cells;

      TransferData<dim, number> transfer_data{gpu_data_coarse,
                                              gpu_data_fine,
                                              weights_view_kokkos[color],
                                              prolongation_matrix_1d,
                                              cell_lists_fine_to_coarse[color],
                                              boundary_dofs_mask_coarse[color],
                                              boundary_dofs_mask_fine[color]};

      Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>
          team_policy(exec, n_cells, Kokkos::AUTO);

      CellProlongationKernel<dim, p_coarse, p_fine, number> prolongator(
          transfer_data, src, dst);

      Kokkos::parallel_for("prolongate_" + std::to_string(color), team_policy,
                           prolongator);
    }
    dst.compress(VectorOperation::add);
  }
  src.zero_out_ghost_values();

  Assert(
      dst.get_partitioner() == matrix_free_fine->get_vector_partitioner(),
      ExcMessage("Fine vector is not handled correclty after prolongation."));

  Assert(
      src.get_partitioner() == matrix_free_coarse->get_vector_partitioner(),
      ExcMessage("Coarse vector is not handled correclty after prolongation."));
}

template <int dim, int p_coarse, int p_fine, typename number,
          bool overlap_communication_computation>
void PolynomialTransfer<dim, p_coarse, p_fine, number,
                        overlap_communication_computation>::
    restrict_and_add(
        LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
        const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
            &src) const {
  Assert(dst.get_partitioner() == matrix_free_coarse->get_vector_partitioner(),
         ExcMessage("Coarse vector is not initialized correctly."));

  Assert(src.get_partitioner() == matrix_free_fine->get_vector_partitioner(),
         ExcMessage("Fine vector is not initialized correctly."));

  MemorySpace::Default::kokkos_space::execution_space exec;

  const auto &colored_graph = matrix_free_fine->get_colored_graph();
  const unsigned int n_colors = colored_graph.size();

  if (overlap_communication_computation) {
    auto do_color = [&](const unsigned int color) {
      const auto &gpu_data_coarse = matrix_free_coarse->get_data(0, color);
      const auto &gpu_data_fine = matrix_free_fine->get_data(0, color);

      const auto n_cells = gpu_data_fine.n_cells;

      Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>
          team_policy(exec, n_cells, Kokkos::AUTO);

      TransferData<dim, number> transfer_data{gpu_data_coarse,
                                              gpu_data_fine,
                                              weights_view_kokkos[color],
                                              prolongation_matrix_1d,
                                              cell_lists_fine_to_coarse[color],
                                              boundary_dofs_mask_coarse[color],
                                              boundary_dofs_mask_fine[color]};

      CellRestrictionKernel<dim, p_coarse, p_fine, number> restrictor(
          transfer_data, src, dst);

      Kokkos::parallel_for("restrict_" + std::to_string(color), team_policy,
                           restrictor);
    };

    src.update_ghost_values_start(0);

    if (n_colors > 0 && colored_graph[0].size() > 0)
      do_color(0);

    src.update_ghost_values_finish();

    if (n_colors > 1 && colored_graph[1].size() > 0) {
      do_color(1);

      // We need a synchronization point because we don't want
      // device-aware MPI to start the MPI communication until the
      // kernel is done.
      Kokkos::fence();
    }
    dst.compress_start(0, VectorOperation::add);

    if (n_colors > 2 && colored_graph[2].size() > 0)
      do_color(2);

    dst.compress_finish(VectorOperation::add);
  } else {
    src.update_ghost_values();

    for (unsigned int color = 0; color < n_colors; ++color) {
      const auto &gpu_data_coarse = matrix_free_coarse->get_data(0, color);
      const auto &gpu_data_fine = matrix_free_fine->get_data(0, color);

      const auto n_cells = gpu_data_fine.n_cells;

      TransferData<dim, number> transfer_data{gpu_data_coarse,
                                              gpu_data_fine,
                                              weights_view_kokkos[color],
                                              prolongation_matrix_1d,
                                              cell_lists_fine_to_coarse[color],
                                              boundary_dofs_mask_coarse[color],
                                              boundary_dofs_mask_fine[color]};

      Kokkos::TeamPolicy<MemorySpace::Default::kokkos_space::execution_space>
          team_policy(exec, n_cells, Kokkos::AUTO);

      CellRestrictionKernel<dim, p_coarse, p_fine, number> restrictor(
          transfer_data, src, dst);

      Kokkos::parallel_for("restrict_" + std::to_string(color), team_policy,
                           restrictor);
    }
    dst.compress(VectorOperation::add);
  }
  src.zero_out_ghost_values();

  Assert(
      dst.get_partitioner() == matrix_free_coarse->get_vector_partitioner(),
      ExcMessage("Coarse vector is not handled correclty after restrtiction."));

  Assert(
      src.get_partitioner() == matrix_free_fine->get_vector_partitioner(),
      ExcMessage("Fine vector is not handled correclty after restrtiction."));
}

template <int dim, int p_coarse, int p_fine, typename number,
          bool overlap_communication_computation>
void PolynomialTransfer<dim, p_coarse, p_fine, number,
                        overlap_communication_computation>::
    reinit(const MatrixFree<dim, number> &mf_coarse,
           const MatrixFree<dim, number> &mf_fine,
           const AffineConstraints<number> &constraints_coarse,
           const AffineConstraints<number> &constraints_fine) {
  this->matrix_free_coarse = &mf_coarse;
  this->matrix_free_fine = &mf_fine;

  this->constraints_coarse = &constraints_coarse;
  this->constraints_fine = &constraints_fine;

  auto &colored_graph_coarse = this->matrix_free_coarse->get_colored_graph();

  const auto &colored_graph_fine = this->matrix_free_fine->get_colored_graph();

  const unsigned int n_colors = colored_graph_fine.size();

  Assert(
      n_colors == colored_graph_coarse.size(),
      ExcMessage("Coarse and fine levels must have the same number of colors"));

  FE_Q<1> fe_coarse_1d(p_coarse);
  FE_Q<1> fe_fine_1d(p_fine);

  std::vector<unsigned int> renumbering_fine(fe_fine_1d.n_dofs_per_cell());

  renumbering_fine[0] = 0;
  for (unsigned int i = 0; i < fe_fine_1d.dofs_per_line; ++i)
    renumbering_fine[i + fe_fine_1d.n_dofs_per_vertex()] =
        GeometryInfo<1>::vertices_per_cell * fe_fine_1d.n_dofs_per_vertex() + i;

  if (fe_fine_1d.n_dofs_per_vertex() > 0)
    renumbering_fine[fe_fine_1d.n_dofs_per_cell() -
                     fe_fine_1d.n_dofs_per_vertex()] =
        fe_fine_1d.n_dofs_per_vertex();

  std::vector<unsigned int> renumbering_coarse(fe_coarse_1d.n_dofs_per_cell());

  renumbering_coarse[0] = 0;
  for (unsigned int i = 0; i < fe_coarse_1d.dofs_per_line; ++i)
    renumbering_coarse[i + fe_coarse_1d.n_dofs_per_vertex()] =
        GeometryInfo<1>::vertices_per_cell * fe_coarse_1d.n_dofs_per_vertex() +
        i;

  if (fe_coarse_1d.n_dofs_per_vertex() > 0)
    renumbering_coarse[fe_coarse_1d.n_dofs_per_cell() -
                       fe_coarse_1d.n_dofs_per_vertex()] =
        fe_coarse_1d.n_dofs_per_vertex();

  FullMatrix<number> matrix(fe_fine_1d.n_dofs_per_cell(),
                            fe_coarse_1d.n_dofs_per_cell());

  FETools::get_projection_matrix(fe_coarse_1d, fe_fine_1d, matrix);

  this->prolongation_matrix_1d =
      Kokkos::View<number *, MemorySpace::Default::kokkos_space>(
          Kokkos::view_alloc("prolongation_matrix_1d_" +
                                 std::to_string(p_coarse) + "_to_" +
                                 std::to_string(p_fine),
                             Kokkos::WithoutInitializing),
          fe_coarse_1d.n_dofs_per_cell() * fe_fine_1d.n_dofs_per_cell());

  auto prolongation_matrix_1d_view =
      Kokkos::create_mirror_view(this->prolongation_matrix_1d);

  for (unsigned int i = 0, k = 0; i < fe_coarse_1d.n_dofs_per_cell(); ++i)
    for (unsigned int j = 0; j < fe_fine_1d.n_dofs_per_cell(); ++j, ++k)
      prolongation_matrix_1d_view[k] =
          matrix(renumbering_fine[j], renumbering_coarse[i]);

  Kokkos::deep_copy(this->prolongation_matrix_1d, prolongation_matrix_1d_view);
  Kokkos::fence();

  const auto &tria =
      this->matrix_free_coarse->get_dof_handler().get_triangulation();
  std::vector<std::vector<unsigned int>> coarse_cell_ids(n_colors);

  for (unsigned int color = 0; color < n_colors; ++color) {
    coarse_cell_ids[color].resize(tria.n_active_cells());

    const auto &graph = colored_graph_coarse[color];

    auto cell = graph.cbegin(), cell_end = graph.cend();

    for (int cell_id = 0; cell != cell_end; ++cell, ++cell_id)
      coarse_cell_ids[color][(*cell)->active_cell_index()] = cell_id;
  }

  this->cell_lists_fine_to_coarse.clear();
  this->cell_lists_fine_to_coarse.resize(n_colors);

  for (unsigned int color = 0; color < n_colors; ++color) {
    const auto &graph = colored_graph_fine[color];

    this->cell_lists_fine_to_coarse[color] =
        Kokkos::View<int *, MemorySpace::Default::kokkos_space>(
            Kokkos::view_alloc("cell_lists_fine_to_coarse_" +
                                   std::to_string(p_coarse) + "_to_" +
                                   std::to_string(p_fine) + "_color_" +
                                   std::to_string(color),
                               Kokkos::WithoutInitializing),
            graph.size());

    auto cell_list_host_view =
        Kokkos::create_mirror_view(this->cell_lists_fine_to_coarse[color]);

    auto cell = graph.cbegin(), cell_end = graph.cend();

    for (int cell_id = 0; cell != cell_end; ++cell, ++cell_id)
      cell_list_host_view[cell_id] =
          coarse_cell_ids[color][(*cell)->active_cell_index()];

    Kokkos::deep_copy(this->cell_lists_fine_to_coarse[color],
                      cell_list_host_view);
    Kokkos::fence();
  }

  setup_weights_and_boundary_dofs_masks();
}

template <int dim, int p_coarse, int p_fine, typename number,
          bool overlap_communication_computation>
void PolynomialTransfer<dim, p_coarse, p_fine, number,
                        overlap_communication_computation>::
    setup_weights_and_boundary_dofs_masks() {
  const auto &dof_handler_fine = matrix_free_fine->get_dof_handler();
  const auto &dof_handler_coarse = matrix_free_coarse->get_dof_handler();
  const auto &fe_fine = dof_handler_fine.get_fe();
  const auto &fe_coarse = dof_handler_coarse.get_fe();

  const auto &colored_graph_fine = matrix_free_fine->get_colored_graph();
  const auto &colored_graph_coarse = matrix_free_coarse->get_colored_graph();

  const unsigned int n_colors = colored_graph_fine.size();

  Assert(
      n_colors == colored_graph_coarse.size(),
      ExcMessage(
          "Portable matrix free objects must have the same number of colors"));

  const unsigned int n_dofs_per_cell_fine = fe_fine.n_dofs_per_cell();
  const unsigned int n_dofs_per_cell_coarse = fe_coarse.n_dofs_per_cell();

  std::vector<unsigned int> lex_numbering_fine(n_dofs_per_cell_fine);
  std::vector<unsigned int> lex_numbering_coarse(n_dofs_per_cell_coarse);

  {
    const Quadrature<1> dummy_quadrature(std::vector<Point<1>>(1, Point<1>()));
    dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;

    shape_info.reinit(dummy_quadrature, fe_fine, 0);
    lex_numbering_fine = shape_info.lexicographic_numbering;
  }

  {
    const Quadrature<1> dummy_quadrature(std::vector<Point<1>>(1, Point<1>()));
    dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shape_info;

    shape_info.reinit(dummy_quadrature, fe_coarse, 0);
    lex_numbering_coarse = shape_info.lexicographic_numbering;
  }

  unsigned int n_cells_fine = 0;
  for (const auto &cell : dof_handler_fine.active_cell_iterators())
    if (cell->is_locally_owned())
      ++n_cells_fine;

  dealii::internal::MatrixFreeFunctions::ConstraintInfo<
      dim, VectorizedArray<number>, types::global_dof_index>
      constraint_info_fine;

  constraint_info_fine.reinit(dof_handler_fine, n_cells_fine);

  constraint_info_fine.set_locally_owned_indices(
      dof_handler_fine.locally_owned_dofs());

  std::vector<types::global_dof_index> local_dof_indices_fine(
      n_dofs_per_cell_fine);
  std::vector<types::global_dof_index> local_dof_indices_lex_fine(
      n_dofs_per_cell_fine);

  int cell_counter = 0;

  for (unsigned int color = 0; color < n_colors; ++color)
    for (const auto &cell : colored_graph_fine[color]) {
      cell->get_dof_indices(local_dof_indices_fine);

      for (unsigned int i = 0; i < n_dofs_per_cell_fine; ++i)
        local_dof_indices_lex_fine[i] =
            local_dof_indices_fine[lex_numbering_fine[i]];

      constraint_info_fine.read_dof_indices(cell_counter,
                                            local_dof_indices_lex_fine, {});
      ++cell_counter;
    }

  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_fine =
      constraint_info_fine.finalize(dof_handler_fine.get_mpi_communicator());

  LinearAlgebra::distributed::Vector<number> weight_vector;
  weight_vector.reinit(partitioner_fine);

  for (const auto i : constraint_info_fine.dof_indices)
    weight_vector.local_element(i) += 1.0;

  weight_vector.compress(VectorOperation::add);

  for (unsigned int i = 0; i < weight_vector.locally_owned_size(); ++i)
    if (weight_vector.local_element(i) > 0)
      weight_vector.local_element(i) = 1.0 / weight_vector.local_element(i);

  // ... clear constrained indices
  for (const auto &constrained_dofs : constraints_fine->get_lines())
    if (weight_vector.locally_owned_elements().is_element(
            constrained_dofs.index))
      weight_vector[constrained_dofs.index] = 0.0;

  weight_vector.update_ghost_values();

  weights_view_kokkos.clear();
  weights_view_kokkos.resize(n_colors);

  for (unsigned int color = 0; color < n_colors; ++color) {
    if (colored_graph_fine[color].size() > 0) {
      const auto &mf_data_fine = matrix_free_fine->get_data(0, color);
      const auto &graph = colored_graph_fine[color];

      weights_view_kokkos[color] =
          Kokkos::View<number **, MemorySpace::Default::kokkos_space>(
              Kokkos::view_alloc("weights_" + std::to_string(color),
                                 Kokkos::WithoutInitializing),
              n_dofs_per_cell_fine, mf_data_fine.n_cells);

      auto weights_view_host =
          Kokkos::create_mirror_view(weights_view_kokkos[color]);

      auto cell = graph.cbegin(), end_cell = graph.cend();

      for (unsigned int cell_id = 0; cell != end_cell; ++cell, ++cell_id) {
        (*cell)->get_dof_indices(local_dof_indices_fine);

        for (unsigned int i = 0; i < n_dofs_per_cell_fine; ++i) {
          types::global_dof_index dof_index_lex =
              local_dof_indices_fine[lex_numbering_fine[i]];
          weights_view_host(i, cell_id) = weight_vector[dof_index_lex];
        }
      }
      Kokkos::deep_copy(weights_view_kokkos[color], weights_view_host);
      Kokkos::fence();
    }
  }

  // setup boundary dofs masks
  std::vector<types::global_dof_index> local_dof_indices_coarse(
      n_dofs_per_cell_coarse);

  this->boundary_dofs_mask_coarse.clear();
  this->boundary_dofs_mask_coarse.resize(n_colors);

  for (unsigned int color = 0; color < n_colors; ++color) {
    if (colored_graph_fine[color].size() > 0) {
      const auto &mf_data_coarse = matrix_free_coarse->get_data(0, color);
      ;
      const auto &graph = colored_graph_coarse[color];

      this->boundary_dofs_mask_coarse[color] =
          Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
              Kokkos::view_alloc("boundary_dofs_mask_coarse_" +
                                     std::to_string(color),
                                 Kokkos::WithoutInitializing),
              n_dofs_per_cell_coarse, mf_data_coarse.n_cells);

      auto dofs_mask_host =
          Kokkos::create_mirror_view(this->boundary_dofs_mask_coarse[color]);

      auto cell = graph.cbegin(), end_cell = graph.cend();

      for (unsigned int cell_id = 0; cell != end_cell; ++cell, ++cell_id) {
        (*cell)->get_dof_indices(local_dof_indices_coarse);

        for (unsigned int i = 0; i < n_dofs_per_cell_coarse; ++i) {
          const auto global_dof =
              local_dof_indices_coarse[lex_numbering_coarse[i]];
          if (constraints_coarse->is_constrained(global_dof))
            dofs_mask_host(i, cell_id) = numbers::invalid_unsigned_int;
          else
            dofs_mask_host(i, cell_id) = global_dof;
        }
      }
      Kokkos::deep_copy(this->boundary_dofs_mask_coarse[color], dofs_mask_host);
      Kokkos::fence();
    }
  }

  this->boundary_dofs_mask_fine.clear();
  this->boundary_dofs_mask_fine.resize(n_colors);

  for (unsigned int color = 0; color < n_colors; ++color) {
    if (colored_graph_fine[color].size() > 0) {
      const auto &mf_data_fine = matrix_free_fine->get_data(0, color);
      const auto &graph = colored_graph_fine[color];

      this->boundary_dofs_mask_fine[color] =
          Kokkos::View<unsigned int **, MemorySpace::Default::kokkos_space>(
              Kokkos::view_alloc("boundary_dofs_mask_fine_" +
                                     std::to_string(color),
                                 Kokkos::WithoutInitializing),
              n_dofs_per_cell_fine, mf_data_fine.n_cells);

      auto dofs_mask_host =
          Kokkos::create_mirror_view(this->boundary_dofs_mask_fine[color]);

      auto cell = graph.cbegin(), end_cell = graph.cend();

      for (unsigned int cell_id = 0; cell != end_cell; ++cell, ++cell_id) {
        (*cell)->get_dof_indices(local_dof_indices_fine);

        for (unsigned int i = 0; i < n_dofs_per_cell_fine; ++i) {
          const auto global_dof = local_dof_indices_fine[lex_numbering_fine[i]];
          if (constraints_fine->is_constrained(global_dof))
            dofs_mask_host(i, cell_id) = numbers::invalid_unsigned_int;
          else
            dofs_mask_host(i, cell_id) = global_dof;
        }
      }
      Kokkos::deep_copy(this->boundary_dofs_mask_fine[color], dofs_mask_host);
      Kokkos::fence();
    }
  }
}

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif