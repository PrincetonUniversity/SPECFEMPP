.. _config_strings:

CONFIG_STRINGS
==============

Overview
--------

The ``CONFIG_STRINGS`` defines a macro for managing configuration strings in SPECFEMPP. It allows developers to define a set of canonical configuration options along with their aliases, enabling flexible input handling and consistent string processing across the application.

Syntax
------

.. code-block:: cpp

   #define CONFIG_STRINGS \
    ((primary_name, alias1, alias2, ...)) \
    ((primary_name2, alias1, alias2, ...)) \
    ...

Where ``primary_name`` is the canonical name for a configuration option, and ``alias1``, ``alias2``, etc. are alternative names that should be recognized as equivalent.

Description
-----------

For each defined string group like ``((hdf5, h5))``, the following functions are generated:

+---------------------------------------+-------------------------------------------------------+
| Function                              | Description                                           |
+=======================================+=======================================================+
| ``is_<primary_name>(std::string)``    | Returns true if the string matches any alias          |
| e.g., ``is_hdf5(str)``                |                                                       |
+---------------------------------------+-------------------------------------------------------+
| ``to_<primary_name>(std::string)``    | Converts any valid alias to the canonical string      |
| e.g., ``to_hdf5(str)``                |                                                       |
+---------------------------------------+-------------------------------------------------------+
| ``from_<primary_name>(std::string)``  | Validates that the string is a valid alias and        |
| e.g., ``from_hdf5(str)``              | returns the normalized form                           |
+---------------------------------------+-------------------------------------------------------+

Examples
--------

Declaration Example
~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #define CONFIG_STRINGS \
    ((hdf5, h5)) \
    ((ascii, txt)) \
    ((psv, p_sv, p-sv))

Usage Example
~~~~~~~~~~~~~

.. code-block:: cpp

   // Check if a string represents the HDF5 format
   std::string format = "h5";
   if (specfem::utilities::is_hdf5(format)) {
     // Use HDF5 format
   }

   // Convert any valid alias to canonical form
   std::string canonical = specfem::utilities::to_hdf5(format); // Returns "hdf5"

   // Validate input against known aliases
   try {
     std::string validated = specfem::utilities::from_hdf5(user_input);
     // Process validated input
   } catch (const std::invalid_argument& e) {
     // Handle invalid input
   }
