Seismograms
###########

Seismograms section defines the output seismograms format at predefined stations. When not-defined seismograms will be calculated but not written to a file.


Parameter definitions
=======================

**Parameter Name** : ``seismogram``
------------------------------------

**default value** : NULL

**possible values** : [YAML Node]

**documentation** : Define seismogram output configuration

**Parameter Name** : ``seismogram.seismogram-format``
******************************************************

**default value** : None

**possible values** : [ascii]

**documentation** : Type of seismogram format to be written.

1. ascii - :ref:`ASCII` writes calculated seismogram values to seismogram files in string format.

**Parameter Name** : ``seismogram.output-folder``
******************************************************

**default value** : Current working directory

**possible values** : [string]

**documentation** : Path to output folder where the seismograms will be saved.
