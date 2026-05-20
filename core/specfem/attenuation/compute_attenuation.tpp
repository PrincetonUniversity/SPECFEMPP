



void compute_attenuation(stress, gradient_pack, point_property, point_attenuation, field_derivatives, assembly.attenuation, assembly.field_derivative_storage) {
    return if attenuation_tag == specfem::element::attenuation_tag::none;

    PointFieldDerivatives old_fd;
    load_on_device(index, assembly.field_derivative_storage, old_fd);
    load_on_device(index, assembly.attenuation, point_attenuation);

    const auto du = gradientpack.get_du<PointTags>();
    const auto dv = gradientpack.get_dv<PointTags>();

    add_relaxation_to_stress(stress, point_attenuation);
    integrate_memory_variables(point_attenuation, du, dv);

    PointFieldDerivatives new_fd =
}
