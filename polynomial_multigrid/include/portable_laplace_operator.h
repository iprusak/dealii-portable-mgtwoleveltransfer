#ifndef portable_laplace_operator_h
#define portable_laplace_operator_h

#include "portable_laplace_operator_base.h"
#include <deal.II/matrix_free/portable_fe_evaluation.h>

DEAL_II_NAMESPACE_OPEN

namespace Portable {

template <int dim, int fe_degree, typename number> class LaplaceOperatorQuad {
public:
  DEAL_II_HOST_DEVICE void
  operator()(FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> *fe_eval,
             const int q_point) const;

  static const unsigned int n_q_points = Utilities::pow(fe_degree + 1, dim);

  static const unsigned int n_local_dofs = n_q_points;
};

template <int dim, int fe_degree, typename number>
DEAL_II_HOST_DEVICE void
LaplaceOperatorQuad<dim, fe_degree, number>::operator()(
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> *fe_eval,
    const int q_point) const {
  fe_eval->submit_gradient(fe_eval->get_gradient(q_point), q_point);
}

template <int dim, int fe_degree, typename number> class LocalLaplaceOperator {
public:
  static constexpr unsigned int n_local_dofs =
      Utilities::pow(fe_degree + 1, dim);
  static constexpr unsigned int n_q_points = Utilities::pow(fe_degree + 1, dim);

  DEAL_II_HOST_DEVICE void
  operator()(const typename MatrixFree<dim, number>::Data *data,
             const DeviceVector<number> &src, DeviceVector<number> &dst) const;
};

template <int dim, int fe_degree, typename number>
DEAL_II_HOST_DEVICE void
LocalLaplaceOperator<dim, fe_degree, number>::operator()(
    const typename MatrixFree<dim, number>::Data *data,
    const DeviceVector<number> &src, DeviceVector<number> &dst) const {
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> fe_eval(data);

  fe_eval.read_dof_values(src);
  fe_eval.evaluate(EvaluationFlags::gradients);
  fe_eval.apply_for_each_quad_point(
      LaplaceOperatorQuad<dim, fe_degree, number>());
  fe_eval.integrate(EvaluationFlags::gradients);
  fe_eval.distribute_local_to_global(dst);
}

template <int dim, int fe_degree, typename number,
          bool overlap_communication_computation>
class LaplaceOperator
    : public LaplaceOperatorBase<dim, number,
                                 overlap_communication_computation> {
public:
  LaplaceOperator(const DoFHandler<dim> &dof_handler,
                  const AffineConstraints<number> &constraints);

  void
  vmult(LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
        const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
            &src) const override;

  void
  Tvmult(LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
         const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
             &src) const override;

  void initialize_dof_vector(
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &vec)
      const override;

  void compute_diagonal() override;

  std::shared_ptr<DiagonalMatrix<
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>>
  get_matrix_diagonal_inverse() const override;

  types::global_dof_index m() const override;

  types::global_dof_index n() const override;

  number el(const types::global_dof_index row,
            const types::global_dof_index col) const override;

  const MatrixFree<dim, number> &get_mf_data() const override;

  const std::shared_ptr<const Utilities::MPI::Partitioner> &
  get_vector_partitioner() const override;

private:
  using TeamHandle = Kokkos::TeamPolicy<
      MemorySpace::Default::kokkos_space::execution_space>::member_type;
  using SharedViewValues = Kokkos::View<
      number **,
      MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using SharedViewGradients = Kokkos::View<
      number ***,
      MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using SharedViewScratchPad = Kokkos::View<
      number *,
      MemorySpace::Default::kokkos_space::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  MatrixFree<dim, number> mf_data;

  static const unsigned int n_q_points = Utilities::pow(fe_degree + 1, dim);

  std::shared_ptr<DiagonalMatrix<
      LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>>
      inverse_diagonal_entries;
};

template <int dim, int fe_degree, typename number,
          bool overlap_communication_computation>
LaplaceOperator<dim, fe_degree, number, overlap_communication_computation>::
    LaplaceOperator(const DoFHandler<dim> &dof_handler,
                    const AffineConstraints<number> &constraints) {
  const MappingQ<dim> mapping(fe_degree);
  typename MatrixFree<dim, number>::AdditionalData additional_data;

  additional_data.mapping_update_flags =
      update_gradients | update_JxW_values | update_quadrature_points;
  additional_data.overlap_communication_computation =
      overlap_communication_computation;

  const QGauss<1> quadrature_1d(fe_degree + 1);
  mf_data.reinit(mapping, dof_handler, constraints, quadrature_1d,
                 additional_data);
}

template <int dim, int fe_degree, typename number,
          bool overlap_communication_computation>
void LaplaceOperator<dim, fe_degree, number,
                     overlap_communication_computation>::
    vmult(LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
          const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
              &src) const {
  AssertDimension(dst.size(), src.size());
  Assert(dst.get_partitioner() == mf_data.get_vector_partitioner(),
         ExcMessage("Vector is not correctly initialized."));
  Assert(src.get_partitioner() == mf_data.get_vector_partitioner(),
         ExcMessage("Vector is not correctly initialized."));

  dst = 0.;
  LocalLaplaceOperator<dim, fe_degree, number> local_operator;

  mf_data.cell_loop(local_operator, src, dst);

  mf_data.copy_constrained_values(src, dst);
}

template <int dim, int fe_degree, typename number,
          bool overlap_communication_computation>
void LaplaceOperator<dim, fe_degree, number,
                     overlap_communication_computation>::
    Tvmult(
        LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &dst,
        const LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
            &src) const {
  AssertDimension(dst.size(), src.size());
  Assert(dst.get_partitioner() == mf_data.get_vector_partitioner(),
         ExcMessage("Vector is not correctly initialized."));
  Assert(src.get_partitioner() == mf_data.get_vector_partitioner(),
         ExcMessage("Vector is not correctly initialized."));

  vmult(dst, src);
}

template <int dim, int fe_degree, typename number,
          bool overlap_communication_computation>
void LaplaceOperator<dim, fe_degree, number,
                     overlap_communication_computation>::
    initialize_dof_vector(
        LinearAlgebra::distributed::Vector<number, MemorySpace::Default> &vec)
        const {
  mf_data.initialize_dof_vector(vec);
}

template <int dim, int fe_degree, typename number,
          bool overlap_communication_computation>
const MatrixFree<dim, number> &
LaplaceOperator<dim, fe_degree, number,
                overlap_communication_computation>::get_mf_data() const {
  return mf_data;
}

template <int dim, int fe_degree, typename number,
          bool overlap_communication_computation>
void LaplaceOperator<dim, fe_degree, number,
                     overlap_communication_computation>::compute_diagonal() {
  this->inverse_diagonal_entries.reset(
      new DiagonalMatrix<
          LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>());

  LinearAlgebra::distributed::Vector<number, MemorySpace::Default>
      &inverse_diagonal = inverse_diagonal_entries->get_vector();
  initialize_dof_vector(inverse_diagonal);

  LaplaceOperatorQuad<dim, fe_degree, number> operator_quad;

  MatrixFreeTools::compute_diagonal<dim, fe_degree, fe_degree + 1, 1, number>(
      mf_data, inverse_diagonal, operator_quad, EvaluationFlags::gradients,
      EvaluationFlags::gradients);

  number *raw_diagonal = inverse_diagonal.get_values();

  Kokkos::parallel_for(
      inverse_diagonal.locally_owned_size(), KOKKOS_LAMBDA(int i) {
        Assert(raw_diagonal[i] > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        raw_diagonal[i] = 1. / raw_diagonal[i];
      });
}

template <int dim, int fe_degree, typename number,
          bool overlap_communication_computation>
std::shared_ptr<DiagonalMatrix<
    LinearAlgebra::distributed::Vector<number, MemorySpace::Default>>>
LaplaceOperator<dim, fe_degree, number, overlap_communication_computation>::
    get_matrix_diagonal_inverse() const {
  return inverse_diagonal_entries;
}

template <int dim, int fe_degree, typename number,
          bool overlap_communication_computation>
types::global_dof_index
LaplaceOperator<dim, fe_degree, number, overlap_communication_computation>::m()
    const {
  return mf_data.get_vector_partitioner()->size();
}

template <int dim, int fe_degree, typename number,
          bool overlap_communication_computation>
types::global_dof_index
LaplaceOperator<dim, fe_degree, number, overlap_communication_computation>::n()
    const {
  return mf_data.get_vector_partitioner()->size();
}

template <int dim, int fe_degree, typename number,
          bool overlap_communication_computation>
number
LaplaceOperator<dim, fe_degree, number, overlap_communication_computation>::el(
    const types::global_dof_index row,
    const types::global_dof_index col) const {
  (void)col;
  Assert(row == col, ExcNotImplemented());
  Assert(inverse_diagonal_entries.get() != nullptr &&
             inverse_diagonal_entries->m() > 0,
         ExcNotInitialized());

  return 1.0 / (*inverse_diagonal_entries)(row, row);
}

template <int dim, int fe_degree, typename number,
          bool overlap_communication_computation>
const std::shared_ptr<const Utilities::MPI::Partitioner> &
LaplaceOperator<dim, fe_degree, number,
                overlap_communication_computation>::get_vector_partitioner()
    const {
  return mf_data.get_vector_partitioner();
}

} // namespace Portable

DEAL_II_NAMESPACE_CLOSE

#endif
