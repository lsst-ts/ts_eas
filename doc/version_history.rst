.. py:currentmodule:: lsst.ts.eas

.. _lsst.ts.eas.version_history:

###############
Version History
###############

v0.6.0
======

* Add Brian Stadler's script for M1M3 thermal control.

v0.5.1
======

* Update the version of ts-conda-build to 0.4 in the conda recipe.

Requires:

* ts_salobj 7.0
* ts_idl 3.1
* IDL file for EAS from ts_xml 8.0

v0.5.0
======

* Improve entry point.
* Add support for multiple Python versions for conda.
* Sort imports with isort.
* Install new pre-commit hooks.
* Add MyPy support.

Requires:

* ts_salobj 7.0
* ts_idl 3.1
* IDL file for EAS from ts_xml 8.0

v0.4.0
======

* Modernize pre-commit config versions.
* Switch to pyproject.toml.
* Use entry_points instead of bin scripts.

Requires:

* ts_salobj 7.0
* ts_idl 3.1
* IDL file for EAS from ts_xml 8.0

v0.3.0
======

Prepared for salobj 7.

Requires:

* ts_salobj 7.0
* ts_idl 3.1
* IDL file for EAS from ts_xml 8.0

v0.2.0
======

Upgraded black to version 20 and ts-conda-build to 0.3

Requires:

* ts_salobj 6.3
* ts_idl 3.1
* IDL file for EAS from ts_xml 8.0


v0.1.0
======

First release of the EAS CSC.

This version basically is an empty CSC to which functionality will be added in a later stage.

Requires:

* ts_salobj 6.3
* ts_idl 3.0
* IDL file for EAS from ts_xml 8.0
