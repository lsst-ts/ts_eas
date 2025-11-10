# This file is part of ts_eas.
#
# Developed for the Vera C. Rubin Observatory Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["RemoteManager"]

from typing import ClassVar

from async_lru import alru_cache
from lsst.ts import salobj


class RemoteManager:
    """Manage and cache SAL Remote instances with a shared Domain.

    The Domain must be initialized once via `initialize(domain)` before
    any calls to `get_remote`.

    Remotes are cached by (name, index) using a decorator on `get_remote`.
    """

    _domain: ClassVar[salobj.Domain | None] = None
    _salinfo_copy: ClassVar[salobj.SalInfo | None] = None
    _remotes: ClassVar[set[salobj.Remote]] = set()  # Needed for pytest cleanup.

    @classmethod
    def initialize(cls, domain: salobj.Domain) -> None:
        """Initialize the shared Domain for all Remotes.

        Parameters
        ----------
        domain : `~salobj.Domain`
            The Domain used to construct all Remote instances.

        Raises
        ------
        RuntimeError
            If called with a different Domain after initialization.
        """
        if cls._domain is not None and cls._domain is not domain:
            raise RuntimeError(
                "RemoteManager already initialized with a different Domain."
            )
        cls._domain = domain

    @classmethod
    @alru_cache(maxsize=None)
    async def get_remote(cls, name: str, index: int = 0) -> salobj.Remote:
        """Return a cached Remote for (name, index), constructing if needed.

        Parameters
        ----------
        name : `str`
            CSC name, e.g., "ESS", "HVAC".
        index : `int`
            CSC index.

        Returns
        -------
        remote : `~salobj.Remote`
            Cached or newly constructed Remote.

        Raises
        ------
        RuntimeError
            If the Domain has not been initialized.
        """
        if cls._domain is None:
            raise RuntimeError("RemoteManager not initialized with a Domain.")
        remote = salobj.Remote(domain=cls._domain, name=name, index=index)
        await remote.start_task
        cls._remotes.add(remote)
        return remote

    @classmethod
    @alru_cache(maxsize=None)
    async def apply_setpoints_topic(cls) -> salobj.topics.ReadTopic:
        """Returns a MTM1M3TS.applySetpoints read topic."""
        if cls._domain is None:
            raise RuntimeError("RemoteManager not initialized with a Domain.")

        cls._salinfo_copy = salobj.SalInfo(cls._domain, "MTM1M3TS")
        apply_setpoints_topic = salobj.topics.ReadTopic(
            salinfo=cls._salinfo_copy, attr_name="cmd_applySetpoints", max_history=1
        )
        await cls._salinfo_copy.start()
        return apply_setpoints_topic

    @classmethod
    async def reset(cls) -> None:
        """Restore the RemoteManager to its initial uninitialized state.

        This method:
        - Closes any active Remotes.
        - Clears all cached return values.
        - Resets the shared Domain and SalInfo references.
        """
        # Attempt to close cached Remotes gracefully
        for remote in cls._remotes:
            await remote.close()
        cls.get_remote.cache_clear()

        # Clear cached apply_setpoints_topic
        cls.apply_setpoints_topic.cache_clear()

        # Attempt to stop SalInfo if active
        if cls._salinfo_copy is not None:
            await cls._salinfo_copy.close()
            cls._salinfo_copy = None

        # Reset the Domain reference
        cls._domain = None
