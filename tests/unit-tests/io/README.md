# Note on the sources and receiver tests

We implemented read_yaml_sources and read_yaml_receivers tests only to cover
the parsing of YAML specfem_config dictionaries from python. If we ever change
how the Python API parses sources and receivers to the cpp code, the tests and
io implementation are likely to require updates or deprecation.
