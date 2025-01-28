from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kfactory.instance import ProtoInstance
    from kfactory.kcell import ProtoKCell
    from kfactory.layer import LayerEnum
    from kfactory.layout import KCLayout
    from kfactory.port import ProtoPort


class LockedError(AttributeError):
    """Raised when a locked cell is being modified."""

    def __init__(self, kcell: ProtoKCell[Any]) -> None:
        """Throw _locked error."""
        super().__init__(
            f"{kcell.name!r} is locked and likely stored in cache. Modifications are "
            "disabled as its associated function is decorated with `cell`. To modify, "
            "update the code in the function or create a copy of "
            f"the {kcell.__class__.__name__}."
        )


class MergeError(ValueError):
    """Raised if two layout's have conflicting cell definitions."""


class PortWidthMismatch(ValueError):
    """Error thrown when two ports don't have a matching `width`."""

    def __init__(
        self,
        inst: ProtoInstance[Any],
        other_inst: ProtoInstance[Any] | ProtoPort[Any],
        p1: ProtoPort[Any],
        p2: ProtoPort[Any],
        *args: Any,
    ) -> None:
        """Throw error for the two ports `p1`/`p1`."""
        if isinstance(other_inst, ProtoInstance):
            super().__init__(
                f'Width mismatch between the ports {inst.cell_name}["{p1.name}"] '
                f'and {other_inst.cell_name}["{p2.name}"]'
                f'("{p1.width}"/"{p2.width}")',
                *args,
            )
        else:
            super().__init__(
                f'Width mismatch between the ports {inst.cell_name}["{p1.name}"] '
                f'and Port "{p2.name}" ("{p1.width}"/"{p2.width}")',
                *args,
            )


class PortLayerMismatch(ValueError):
    """Error thrown when two ports don't have a matching `layer`."""

    def __init__(
        self,
        kcl: KCLayout,
        inst: ProtoInstance[Any],
        other_inst: ProtoInstance[Any] | ProtoPort[Any],
        p1: ProtoPort[Any],
        p2: ProtoPort[Any],
        *args: Any,
    ) -> None:
        """Throw error for the two ports `p1`/`p1`."""
        l1 = (
            f"{p1.layer.name}({p1.layer.__int__()})"
            if isinstance(p1.layer, LayerEnum)
            else str(kcl.layout.get_info(p1.layer))
        )
        l2 = (
            f"{p2.layer.name}({p2.layer.__int__()})"
            if isinstance(p2.layer, LayerEnum)
            else str(kcl.layout.get_info(p2.layer))
        )
        if isinstance(other_inst, ProtoInstance):
            super().__init__(
                f'Layer mismatch between the ports {inst.cell_name}["{p1.name}"]'
                f' and {other_inst.cell_name}["{p2.name}"] ("{l1}"/"{l2}")',
                *args,
            )
        else:
            super().__init__(
                f'Layer mismatch between the ports {inst.cell_name}["{p1.name}"]'
                f' and Port "{p2.name}" ("{l1}"/"{l2}")',
                *args,
            )


class PortTypeMismatch(ValueError):
    """Error thrown when two ports don't have a matching `port_type`."""

    def __init__(
        self,
        inst: ProtoInstance[Any],
        other_inst: ProtoInstance[Any] | ProtoPort[Any],
        p1: ProtoPort[Any],
        p2: ProtoPort[Any],
        *args: Any,
    ) -> None:
        """Throw error for the two ports `p1`/`p1`."""
        if isinstance(other_inst, ProtoInstance):
            super().__init__(
                f'Type mismatch between the ports {inst.cell_name}["{p1.name}"]'
                f' and {other_inst.cell_name}["{p2.name}"]'
                f" ({p1.port_type}/{p2.port_type})",
                *args,
            )
        else:
            super().__init__(
                f'Type mismatch between the ports {inst.cell_name}["{p1.name}"]'
                f' and Port "{p2.name}" ({p1.port_type}/{p2.port_type})',
                *args,
            )


class CellNameError(ValueError):
    """Raised if a KCell is created and the automatic assigned name is taken."""
