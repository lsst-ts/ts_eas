v0.10.0 (2025-12-12)
===================

New Features
------------

- Added AHU setpoint delta to config. (`OSW-1067 <https://rubinobs.atlassian.net//browse/OSW-1067>`_)
- Added louvers to dome opening condition. (`OSW-1488 <https://rubinobs.atlassian.net//browse/OSW-1488>`_)


Bug Fixes
---------

- Fixed deadband logic in TMA setpoints. (`OSW-1563 <https://rubinobs.atlassian.net//browse/OSW-1563>`_)


v0.9.7 (2025-12-04)
===================

Bug Fixes
---------

- Fixed detection of dome open condition. (`OSW-1540 <https://rubinobs.atlassian.net//browse/OSW-1540>`_`)


v0.9.6 (2025-11-27)
===================

Bug Fixes
---------

- Removed non-existent 'index' attribute of Remote. (`OSW-1513 <https://rubinobs.atlassian.net//browse/OSW-1513>`_)


v0.9.5 (2025-11-21)
===================

New Features
------------

- Ensured Remote objects are constructed one time only. (`OSW-1405 <https://rubinobs.atlassian.net//browse/OSW-1405>`_)


v0.9.4 (2025-11-13)
===================

Bug Fixes
---------

- Changed control of VEC-04 to device ID 802. (`OSW-1409 <https://rubinobs.atlassian.net//browse/OSW-1409>`_)


v0.9.3 (2025-11-07)
===================

New Features
------------

- Added glycol_chillers as a features_to_disable option. (`OSW-1391 <https://rubinobs.atlassian.net//browse/OSW-1391>`_)


v0.9.2 (2025-11-04)
===================

New Features
------------

- Made fan speed parameters configurable. (`OSW-1147 <https://rubinobs.atlassian.net//browse/OSW-1147>`_)


Performance Enhancement
-----------------------

- Updated ts-conda-build dependency version and conda build string. (`OSW-1207 <https://rubinobs.atlassian.net//browse/OSW-1207>`_)


v0.9.1 (2025-09-04)
===================

Bug Fixes
---------

- Added headerPWM argument to the heaterFanDemand command. (`OSW-996 <https://rubinobs.atlassian.net//browse/OSW-996>`_)


v0.9.0 (2025-09-01)
===================

New Features
------------

- Added HVAC setpoint control for nighttime. (`OSW-656 <https://rubinobs.atlassian.net//browse/OSW-656>`_)
- Added control of the M1M3TS fans and dynamic glycol setpoint offset. (`OSW-820 <https://rubinobs.atlassian.net//browse/OSW-820>`_)
- Rely on SAL to get the previous MTM1M3TS setpoint. (`OSW-878 <https://rubinobs.atlassian.net//browse/OSW-878>`_)


Bug Fixes
---------

- Don't skip last night in searching for twilight temperature. (`OSW-878 <https://rubinobs.atlassian.net//browse/OSW-878>`_)


v0.8.6 (2025-08-12)
===================

New Features
------------

- Implemented cooling rate limit. (`OSW-752 <https://rubinobs.atlassian.net//browse/OSW-752>`_)


Bug Fixes
---------

- Fixed MTMount.setThermal to supply state argument. (`OSW-752 <https://rubinobs.atlassian.net//browse/OSW-752>`_)


v0.8.5 (2025-07-23)
===================

Bug Fixes
---------

* Corrected application of setpoint to M1M3TS when the temperature is warming.

New Features
------------

* Added lower limit for HVAC temperature.
* Added control of the top end setpoint.


v0.8.2 (2025-06-13)
===================

New Features
------------

- * Added noon and twilight timers.
  * Added control of the AHU setpoint at noon.
  * Added M1M3TS control. (`DM-50705 <https://rubinobs.atlassian.net//browse/DM-50705>`_)
- * Sets up M1M3TS to track ESS:112 when the dome is open.
  * Deadbands and heating limits added for MTM1M3TS.
  * Several parameters added to configuration. (`DM-51013 <https://rubinobs.atlassian.net//browse/DM-51013>`_)


v0.8.1 (2025-05-20)
===================

Bug Fixes
---------

- Fixed memory leak. (`DM-51001 <https://rubinobs.atlassian.net//browse/DM-51001>`_)


v0.8.0 (2025-05-16)
===================

New Features
------------

- Added control loop for the HVAC based on dome and wind state. (`DM-50351 <https://rubinobs.atlassian.net//browse/DM-50351>`_)
- Added towncrier. (`DM-50624 <https://rubinobs.atlassian.net//browse/DM-50624>`_)


API Removal or Deprecation
--------------------------

- Removed M1M3TS control loop. (`DM-50624 <https://rubinobs.atlassian.net//browse/DM-50624>`_)


v0.6.1
======

* Implemented fans and valve off on CSC disable.

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
