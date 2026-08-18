#pragma once

#include "specfem/assembly/attenuation.hpp"
#include "specfem/assembly/attenuation/impl/attenuation_medium.hpp"
#include "specfem/datatype/domain_view.hpp"
#include "specfem/element.hpp"
#include "specfem/setup.hpp"
#include "specfem/tag_dispatch.hpp"
#include <Kokkos_Core.hpp>
#include <tuple>
#include <type_traits>

namespace specfem::solver::impl {

/**
 * @brief True when @p MediumType carries the out-of-plane memory variables that
 * only exist in three dimensions.
 *
 * Two-dimensional attenuation media store three standard-linear-solid memory
 * variables (\f$ R_\kappa, R_{xx}, R_{xz} \f$) and three strain components;
 * three-dimensional media add \f$ R_{yy}, R_{zz}, R_{xy}, R_{yz} \f$ and three
 * more strain components. This concept is the only place the two shapes are
 * distinguished.
 *
 * @tparam MediumType A specfem::assembly::impl::attenuation_medium
 * specialization
 */
template <typename MediumType>
concept has_three_dimensional_memory_variables =
    requires(MediumType medium) { medium.memory_variable_Ryy; };

/**
 * @brief Saved copy of the attenuation state of a single medium.
 *
 * Members mirror the medium's own view names, dropping the @c memory_variable_
 * and @c _att decorations. In two dimensions the out-of-plane members are left
 * default-constructed and never touched.
 *
 * @tparam MediumType The attenuation medium whose views are being saved
 */
template <typename MediumType> struct AttenuationSnapshot {
  using memory_view_type = std::remove_cvref_t<
      decltype(std::declval<MediumType>().memory_variable_kappa)>;
  using strain_view_type =
      std::remove_cvref_t<decltype(std::declval<MediumType>().epsilon_xx_att)>;

  memory_view_type Rkappa;     ///< Bulk memory variable
  memory_view_type Rxx;        ///< Deviatoric memory variable, xx
  memory_view_type Rxz;        ///< Deviatoric memory variable, xz
  memory_view_type Ryy;        ///< Deviatoric memory variable, yy (dim3 only)
  memory_view_type Rzz;        ///< Deviatoric memory variable, zz (dim3 only)
  memory_view_type Rxy;        ///< Deviatoric memory variable, xy (dim3 only)
  memory_view_type Ryz;        ///< Deviatoric memory variable, yz (dim3 only)
  strain_view_type epsilon_xx; ///< Attenuation strain, xx
  strain_view_type epsilon_zz; ///< Attenuation strain, zz
  strain_view_type epsilon_xz; ///< Attenuation strain, xz
  strain_view_type epsilon_yy; ///< Attenuation strain, yy (dim3 only)
  strain_view_type epsilon_xy; ///< Attenuation strain, xy (dim3 only)
  strain_view_type epsilon_yz; ///< Attenuation strain, yz (dim3 only)
};

/**
 * @brief One attenuation view, named both on the live medium and in a snapshot.
 *
 * @tparam MediumMember   Pointer-to-member type on the attenuation medium
 * @tparam SnapshotMember Pointer-to-member type on the snapshot
 */
template <typename MediumMember, typename SnapshotMember>
struct AttenuationViewPair {
  MediumMember on_medium;     ///< The live view the solver integrates
  SnapshotMember in_snapshot; ///< The saved copy of that view
};

/**
 * @brief Build an AttenuationViewPair with deduced member-pointer types.
 *
 * Written as a function rather than relying on aggregate class-template
 * argument deduction, which is not available on every supported host compiler.
 *
 * @tparam MediumMember   Pointer-to-member type on the attenuation medium
 * @tparam SnapshotMember Pointer-to-member type on the snapshot
 * @param on_medium   Pointer to the view member on the attenuation medium
 * @param in_snapshot Pointer to the matching member on the snapshot
 * @return The paired member pointers
 */
template <typename MediumMember, typename SnapshotMember>
constexpr auto make_attenuation_view_pair(MediumMember on_medium,
                                          SnapshotMember in_snapshot) {
  return AttenuationViewPair<MediumMember, SnapshotMember>{ on_medium,
                                                            in_snapshot };
}

/**
 * @brief The authoritative list of attenuation views a replay saves and
 * restores.
 *
 * Every save, restore, refresh and reset in this header walks this one list, so
 * a new memory variable only has to be registered here. Visit order fixes the
 * order of the allocations and copies that follow.
 *
 * @tparam MediumType The attenuation medium whose views are being listed
 * @return Tuple of six pairs in two dimensions, thirteen in three
 */
template <typename MediumType> constexpr auto attenuation_views() {
  using Snapshot = AttenuationSnapshot<MediumType>;

  auto in_plane = std::make_tuple(
      make_attenuation_view_pair(&MediumType::memory_variable_kappa,
                                 &Snapshot::Rkappa),
      make_attenuation_view_pair(&MediumType::memory_variable_Rxx,
                                 &Snapshot::Rxx),
      make_attenuation_view_pair(&MediumType::memory_variable_Rxz,
                                 &Snapshot::Rxz),
      make_attenuation_view_pair(&MediumType::epsilon_xx_att,
                                 &Snapshot::epsilon_xx),
      make_attenuation_view_pair(&MediumType::epsilon_zz_att,
                                 &Snapshot::epsilon_zz),
      make_attenuation_view_pair(&MediumType::epsilon_xz_att,
                                 &Snapshot::epsilon_xz));

  if constexpr (has_three_dimensional_memory_variables<MediumType>) {
    return std::tuple_cat(
        in_plane,
        std::make_tuple(make_attenuation_view_pair(
                            &MediumType::memory_variable_Ryy, &Snapshot::Ryy),
                        make_attenuation_view_pair(
                            &MediumType::memory_variable_Rzz, &Snapshot::Rzz),
                        make_attenuation_view_pair(
                            &MediumType::memory_variable_Rxy, &Snapshot::Rxy),
                        make_attenuation_view_pair(
                            &MediumType::memory_variable_Ryz, &Snapshot::Ryz),
                        make_attenuation_view_pair(&MediumType::epsilon_yy_att,
                                                   &Snapshot::epsilon_yy),
                        make_attenuation_view_pair(&MediumType::epsilon_xy_att,
                                                   &Snapshot::epsilon_xy),
                        make_attenuation_view_pair(&MediumType::epsilon_yz_att,
                                                   &Snapshot::epsilon_yz)));
  } else {
    return in_plane;
  }
}

/**
 * @brief Apply @p visit to every (live view, saved view) pair of one medium.
 *
 * @tparam MediumType   The attenuation medium, const-qualified when it is only
 *                      being read
 * @tparam SnapshotType The matching AttenuationSnapshot, const-qualified when
 *                      it is only being read
 * @tparam Visitor      Callable as `visit(live_view, saved_view)`
 * @param medium   Attenuation medium holding the live views
 * @param snapshot Snapshot holding the saved copies
 * @param visit    Invoked once per view, in attenuation_views() order
 */
template <typename MediumType, typename SnapshotType, typename Visitor>
void for_each_attenuation_view(MediumType &medium, SnapshotType &snapshot,
                               Visitor &&visit) {
  std::apply(
      [&](const auto &...view) {
        (visit(medium.*view.on_medium, snapshot.*view.in_snapshot), ...);
      },
      attenuation_views<std::remove_cvref_t<MediumType>>());
}

/**
 * @brief Apply @p visit to every live attenuation view of one medium.
 *
 * @tparam MediumType The attenuation medium holding the views
 * @tparam Visitor    Callable as `visit(live_view)`
 * @param medium Attenuation medium holding the live views
 * @param visit  Invoked once per view, in attenuation_views() order
 */
template <typename MediumType, typename Visitor>
void for_each_attenuation_view(MediumType &medium, Visitor &&visit) {
  std::apply([&](const auto &...view) { (visit(medium.*view.on_medium), ...); },
             attenuation_views<std::remove_cvref_t<MediumType>>());
}

/**
 * @brief Allocate a same-sized, same-space copy of one attenuation view.
 *
 * @tparam ViewType A specfem::datatype::View over the attenuation domain
 * @param view The view to copy
 * @return A freshly allocated view holding the same values
 */
template <typename ViewType>
ViewType clone_attenuation_view(const ViewType &view) {
  ViewType clone(std::string(view.label()) + "_snapshot", view.get_mapping());
  specfem::datatype::deep_copy(clone, view);
  return clone;
}

/**
 * @brief A saved copy of the whole attenuation state, across every medium.
 *
 * The solver keeps exactly one live attenuation state, which the forward replay
 * and the adjoint sweep take turns owning. This type is how the state that is
 * not currently live gets parked.
 *
 * The three operations are named so that the snapshot is always the subject and
 * the preposition gives the direction of the copy, which makes the two
 * mirror-image operations impossible to confuse:
 *
 * @code
 * auto saved = AttenuationState<dim3>::capture_from(assembly.attenuation);
 * saved.restore_into(assembly.attenuation);  // live  <- saved
 * saved.refresh_from(assembly.attenuation);  // saved <- live
 * @endcode
 *
 * Move-only: copying would silently double the device memory a replay holds,
 * and specfem::tag_dispatch::Storage has an unconstrained initializer
 * constructor that hijacks lvalue copies, so the deleted copy here turns a
 * confusing template error into a clear one.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 */
template <specfem::element::dimension_tag DimensionTag> class AttenuationState {
public:
  using AttenuationType = specfem::assembly::Attenuation<DimensionTag>;

  AttenuationState() = default;

  AttenuationState(const AttenuationState &) = delete;
  AttenuationState &operator=(const AttenuationState &) = delete;
  AttenuationState(AttenuationState &&) = default;
  AttenuationState &operator=(AttenuationState &&) = default;

  /**
   * @brief Allocate a snapshot and fill it with the current live state.
   *
   * @param attenuation The live attenuation state to copy
   * @return A snapshot owning its own copy of every view
   */
  static AttenuationState capture_from(const AttenuationType &attenuation) {
    AttenuationState state;
    state.storage_ = StorageType([&]<typename TagsType>() {
      const auto &medium =
          attenuation.attenuation_storage.template get<TagsType>();
      AttenuationSnapshot<MediumFor<TagsType>> snapshot;
      specfem::solver::impl::for_each_attenuation_view(
          medium, snapshot, [](const auto &live, auto &saved) {
            saved = specfem::solver::impl::clone_attenuation_view(live);
          });
      return snapshot;
    });
    return state;
  }

  /**
   * @brief Copy the saved state back over the live one.
   *
   * @param attenuation The live attenuation state to overwrite
   */
  void restore_into(AttenuationType &attenuation) const {
    specfem::tag_dispatch::for_each(
        AttenuationType::attenuation_medium_combinations,
        [&]<typename TagsType>() {
          auto &medium =
              attenuation.attenuation_storage.template get<TagsType>();
          specfem::solver::impl::for_each_attenuation_view(
              medium, storage_.template get<TagsType>(),
              [](auto &live, const auto &saved) {
                specfem::datatype::deep_copy(live, saved);
              });
        });
  }

  /**
   * @brief Overwrite the saved state with the current live one.
   *
   * @param attenuation The live attenuation state to read
   */
  void refresh_from(const AttenuationType &attenuation) {
    specfem::tag_dispatch::for_each(
        AttenuationType::attenuation_medium_combinations,
        [&]<typename TagsType>() {
          const auto &medium =
              attenuation.attenuation_storage.template get<TagsType>();
          specfem::solver::impl::for_each_attenuation_view(
              medium, storage_.template get<TagsType>(),
              [](const auto &live, auto &saved) {
                specfem::datatype::deep_copy(saved, live);
              });
        });
  }

private:
  template <typename TagsType>
  using MediumFor = specfem::assembly::impl::attenuation_medium<
      TagsType::dimension_tag, TagsType::medium_tag, TagsType::property_tag,
      TagsType::attenuation_tag>;

  template <typename TagsType>
  using SnapshotFor = AttenuationSnapshot<MediumFor<TagsType>>;

  using StorageType = specfem::tag_dispatch::TypedStorage<
      SnapshotFor, decltype(AttenuationType::attenuation_medium_combinations)>;

  StorageType storage_;
};

/**
 * @brief Zero every attenuation memory variable and strain component.
 *
 * The adjoint wavefield starts each simulation from a quiescent attenuation
 * state, unlike the forward wavefield, which resumes from a checkpoint.
 *
 * @tparam DimensionTag Spatial dimension (dim2 or dim3)
 * @param attenuation The live attenuation state to clear
 */
template <specfem::element::dimension_tag DimensionTag>
void reset_attenuation_state(
    specfem::assembly::Attenuation<DimensionTag> &attenuation) {
  specfem::tag_dispatch::for_each(
      specfem::assembly::Attenuation<
          DimensionTag>::attenuation_medium_combinations,
      [&]<typename TagsType>() {
        auto &medium = attenuation.attenuation_storage.template get<TagsType>();
        if (medium.element_range.extent(0) == 0)
          return;

        specfem::solver::impl::for_each_attenuation_view(
            medium, [](auto &live) {
              Kokkos::deep_copy(live, static_cast<type_real>(0));
            });
        medium.copy_to_host();
      });
}

} // namespace specfem::solver::impl
