#include "specfem/linear_system/trilinos_smoke.hpp"

#ifdef SPECFEM_ENABLE_TRILINOS

#include <Teuchos_RCP.hpp>
#include <Teuchos_Tuple.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>

int specfem::linear_system::trilinos_smoke_test() {
  using map_type = Tpetra::Map<>;
  // Float scalar: the Trilinos install instantiates single precision only
  // (Tpetra_INST_FLOAT=ON, Tpetra_INST_DOUBLE=OFF), matching type_real = float.
  using crs_matrix_type = Tpetra::CrsMatrix<float>;
  using global_ordinal_type = map_type::global_ordinal_type;
  using scalar_type = crs_matrix_type::scalar_type;

  const auto comm = Tpetra::getDefaultComm();

  const Tpetra::global_size_t num_global_entries = 1;
  const global_ordinal_type index_base = 0;
  const Teuchos::RCP<const map_type> row_map(
      new map_type(num_global_entries, index_base, comm));

  crs_matrix_type matrix(row_map, /*maxNumEntriesPerRow=*/1);
  const global_ordinal_type gid = 0;
  matrix.insertGlobalValues(gid, Teuchos::tuple<global_ordinal_type>(gid),
                            Teuchos::tuple<scalar_type>(scalar_type(1)));
  matrix.fillComplete();

  return static_cast<int>(matrix.getGlobalNumRows());
}

#else

int specfem::linear_system::trilinos_smoke_test() { return 0; }

#endif
