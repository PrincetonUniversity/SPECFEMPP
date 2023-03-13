Seismogram output
==================

On successful completion of the SPECFEM2D run a seismogram will be written to the output directory. Below are definitions output formats as defined in SPECFEM.

.. _stations_file:

STATIONS file
-------------

STATIONS file is tab delimited text file, where every line describes the location of station. Every station location is defined using 6 tab delimited values. For example,

.. code:: bash

    #station_name #network_name #x-position #z-position #elevation #burial depth
    AA            S0001          2500.0      2250.0      0.0        0.0



Seismogram output formats
--------------------------

ASCII format
^^^^^^^^^^^^^

ASCII format is a simple human-readable format for storing seimograms. The seimogram files for statitons are named using the convention ``<network_name><stations_name><component>.<extension>`` , where ``component`` value can be either ``BXX`` (x-component) or ``BXZ`` (z-component) and ``extension`` can be either of ``sema`` (acceleration seismogram), ``semv`` (velocity seimogram) or ``semd`` (displacement seismogram). Every line within the seismogram file defines the seimogram value at a sampling time. An example is shown below:

.. code:: bash

    # time(s)     #seismogram value
    -1.320000e-04 0.000000e+00
    -1.210000e-04 0.000000e+00
    -1.100000e-04 0.000000e+00
    -9.900000e-05 0.000000e+00
    -8.800000e-05 0.000000e+00
    -7.700000e-05 0.000000e+00
    -6.600000e-05 0.000000e+00
    -5.500000e-05 0.000000e+00
    -4.400000e-05 0.000000e+00
    -3.300000e-05 0.000000e+00
    -2.200000e-05 0.000000e+00
    -1.100000e-05 0.000000e+00
    0.000000e+00 2.802597e-45
    1.100000e-05 1.961818e-44
