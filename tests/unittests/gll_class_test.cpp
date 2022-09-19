TEST(gll_class, CHECK_LENGTHS) {
  gll::gll gll_instance = gll::gll();

  gll_instance.set_derivation_matrices();
  Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace> xigll =
      gll_instance.get_xigll();
  EXPECT_EQ(xigll.extent(0), 5);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
  return RUN_ALL_TESTS();
}
