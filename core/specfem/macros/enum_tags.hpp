#pragma once

/**
 * @defgroup material_iterator_enum_tags
 * @brief Macros for enumeration tags used in template metaprogramming
 * @{
 */

/**
 * @brief ID for dimension tags
 *
 * Used to identify dimension tags such as @ref DIMENSION_TAG_DIM2 and @ref
 * DIMENSION_TAG_DIM3. See @ref material_iterator_macros.
 */
#define _ENUM_ID_DIMENSION_TAG 0

/**
 * @brief ID for medium tags
 *
 * Used to identify medium tags such as @ref MEDIUM_TAG_ELASTIC_PSV, @ref
 * MEDIUM_TAG_ACOUSTIC, etc. See @ref material_iterator_macros.
 */
#define _ENUM_ID_MEDIUM_TAG 1

/**
 * @brief ID for property tags
 *
 * Used to identify property tags such as @ref PROPERTY_TAG_ISOTROPIC and @ref
 * PROPERTY_TAG_ANISOTROPIC. See @ref material_iterator_macros.
 */
#define _ENUM_ID_PROPERTY_TAG 2

/**
 * @brief ID for boundary tags
 *
 * Used to identify boundary tags such as @ref BOUNDARY_TAG_NONE and @ref
 * BOUNDARY_TAG_STACEY. See @ref material_iterator_macros.
 */
#define _ENUM_ID_BOUNDARY_TAG 3

/**
 * @brief ID for connection tags
 *
 * Used to identify connection tags in interface definitions.
 * See @ref interface_iterator_macros.
 */
#define _ENUM_ID_CONNECTION_TAG 4

/**
 * @brief ID for interface tags
 *
 * Used to identify interface tags in interface definitions.
 * See @ref interface_iterator_macros.
 */
#define _ENUM_ID_INTERFACE_TAG 5

/**
 * @brief ID for flux scheme tags
 *
 * Used to identify flux scheme tags in interface definitions.
 * See @ref interface_iterator_macros.
 */
#define _ENUM_ID_FLUX_SCHEME_TAG 6

/** @} */ // end of material_iterator_enum_tags
