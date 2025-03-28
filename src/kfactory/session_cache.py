from __future__ import annotations

import functools
import hashlib
import inspect
from collections import defaultdict
from threading import RLock
from typing import TYPE_CHECKING, Annotated, Any, get_origin

from cachetools import Cache, cached
from pydantic import BaseModel

from . import kdb
from .conf import CheckInstances, config, logger
from .exceptions import CellNameError
from .protocols import KCellFunc
from .serialization import (
    DecoratorDict,
    DecoratorList,
    _hashable_to_original,
    _to_hashable,
    get_cell_name,
)
from .settings import KCellSettings, KCellSettingsUnits
from .typings import K, KCellParams, MetaData

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from pathlib import Path

    from cachetools.keys import _HashedTuple  # type: ignore[attr-defined,unused-ignore]

    from .kcell import AnyTKCell
    from .layout import KCLayout


def _file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class FileCache(BaseModel):
    path: Path
    parents: list[Path]
    factories: list[str]

    @property
    def location_hash(self) -> str:
        """Hash of the file's path (location)."""
        return hashlib.sha256(str(self.path).encode("utf-8")).hexdigest()

    @property
    def content_hash(self) -> str:
        """Hash of the file's content."""
        return _file_hash(self.path)


class SessionManger:
    def load_session(self) -> None:
        build_dir = config.root_dir / "build"
        self.session_dir = build_dir / "session"

    def get_factory_cache(self, path_hash: str, file_hash: str) -> kdb.Layout:
        ly = kdb.Layout()
        ly.read(str(config.root_dir / "build" / path_hash / file_hash / "layout.oas"))
        return ly

    # @property
    # def


class WrappedKCellFunc(KCellFunc[KCellParams, K]):
    _f: KCellFunc[KCellParams, K]
    cache: Cache[int, Any] | dict[int, Any]
    name: str | None

    @property
    def __name__(self) -> str:
        if self.name is None:
            raise ValueError(f"{self._f} does not have a name")
        return self.name

    @__name__.setter
    def __name__(self, value: str) -> None:
        self.name = value

    def __init__(
        self,
        *,
        kcl: KCLayout,
        f: KCellFunc[KCellParams, AnyTKCell] | KCellFunc[KCellParams, K],
        sig: inspect.Signature,
        output_type: type[K],
        cache: Cache[int, Any] | dict[int, Any],
        set_settings: bool,
        set_name: bool,
        check_ports: bool,
        check_instances: CheckInstances,
        snap_ports: bool,
        add_port_layers: bool,
        basename: str | None,
        drop_params: Sequence[str],
        overwrite_existing: bool | None,
        layout_cache: bool | None,
        info: dict[str, MetaData] | None,
        post_process: Iterable[Callable[[K], None]],
        debug_names: bool,
    ) -> None:
        @functools.wraps(f)
        def wrapper_autocell(
            *args: KCellParams.args, **kwargs: KCellParams.kwargs
        ) -> K:
            params: dict[str, Any] = {
                p.name: p.default for _, p in sig.parameters.items()
            }
            param_units: dict[str, str] = {
                p.name: p.annotation.__metadata__[0]
                for p in sig.parameters.values()
                if get_origin(p.annotation) is Annotated
            }
            arg_par = list(sig.parameters.items())[: len(args)]
            for i, (k, _) in enumerate(arg_par):
                params[k] = args[i]
            params.update(kwargs)

            del_parameters: list[str] = []

            for key, value in params.items():
                if isinstance(value, dict | list):
                    params[key] = _to_hashable(value)
                elif isinstance(value, kdb.LayerInfo):
                    params[key] = kcl.get_info(kcl.layer(value))
                if value is inspect.Parameter.empty:
                    del_parameters.append(key)

            for param in del_parameters:
                params.pop(param, None)
                param_units.pop(param, None)

            @cached(cache=cache, lock=RLock())
            @functools.wraps(f)
            def wrapped_cell(**params: Any) -> K:
                for key, value in params.items():
                    if isinstance(value, DecoratorDict | DecoratorList):
                        params[key] = _hashable_to_original(value)
                old_future_name: str | None = None
                if set_name:
                    if basename is not None:
                        name = get_cell_name(basename, **params)
                    else:
                        name = get_cell_name(f.__name__, **params)
                    old_future_name = kcl.future_cell_name
                    kcl.future_cell_name = name
                    if layout_cache:
                        if overwrite_existing:
                            for c in list(kcl._cells(kcl.future_cell_name)):
                                kcl[c.cell_index()].delete()
                        else:
                            layout_cell = kcl.layout_cell(kcl.future_cell_name)
                            if layout_cell is not None:
                                logger.debug(
                                    "Loading {} from layout cache",
                                    kcl.future_cell_name,
                                )
                                return kcl.get_cell(
                                    layout_cell.cell_index(), output_type
                                )
                    logger.debug(f"Constructing {kcl.future_cell_name}")
                    name_: str | None = name
                else:
                    name_ = None
                cell = f(**params)  # type: ignore[call-arg]

                logger.debug("Constructed {}", name_ or cell.name)

                if cell.locked:
                    # If the cell is locked, it comes from a cache (most likely)
                    # and should be copied first
                    cell = cell.dup(new_name=kcl.future_cell_name)
                if overwrite_existing:
                    for c in list(kcl._cells(name_ or cell.name)):
                        if c is not cell.kdb_cell:
                            kcl[c.cell_index()].delete()
                if set_name and name_:
                    if debug_names and cell.kcl.layout_cell(name_) is not None:
                        logger.opt(depth=4).error(
                            "KCell with name {name} exists already. Duplicate "
                            "occurrence in module '{module}' at "
                            "line {lno}",
                            name=name_,
                            module=f.__module__,
                            function_name=f.__name__,
                            lno=inspect.getsourcelines(f)[1],
                        )
                        raise CellNameError(f"KCell with name {name_} exists already.")

                    cell.name = name_
                    kcl.future_cell_name = old_future_name
                if set_settings:
                    if hasattr(f, "__name__"):
                        cell.function_name = f.__name__
                    elif hasattr(f, "func"):
                        cell.function_name = f.func.__name__
                    else:
                        raise ValueError(f"Function {f} has no name.")
                    cell.basename = basename

                    for param in drop_params:
                        params.pop(param, None)
                        param_units.pop(param, None)
                    cell.settings = KCellSettings(**params)
                    cell.settings_units = KCellSettingsUnits(**param_units)
                if check_ports:
                    port_names: dict[str | None, int] = defaultdict(int)
                    for port in cell.ports:
                        port_names[port.name] += 1
                    duplicate_names = [
                        (name, n) for name, n in port_names.items() if n > 1
                    ]
                    if duplicate_names:
                        raise ValueError(
                            "Found duplicate port names: "
                            + ", ".join([f"{name}: {n}" for name, n in duplicate_names])
                            + " If this intentional, please pass "
                            "`check_ports=False` to the @cell decorator"
                        )
                match check_instances:
                    case CheckInstances.RAISE:
                        if any(inst.is_complex() for inst in cell.each_inst()):
                            raise ValueError(
                                "Most foundries will not allow off-grid "
                                "instances. Please flatten them or add "
                                "check_instances=False to the decorator.\n"
                                "Cellnames of instances affected by this:"
                                + "\n".join(
                                    inst.cell.name
                                    for inst in cell.each_inst()
                                    if inst.is_complex()
                                )
                            )
                    case CheckInstances.FLATTEN:
                        if any(inst.is_complex() for inst in cell.each_inst()):
                            cell.flatten()
                    case CheckInstances.VINSTANCES:
                        if any(inst.is_complex() for inst in cell.each_inst()):
                            complex_insts = [
                                inst for inst in cell.each_inst() if inst.is_complex()
                            ]
                            for inst in complex_insts:
                                vinst = cell.create_vinst(kcl[inst.cell.cell_index()])
                                vinst.trans = inst.dcplx_trans
                                inst.delete()
                    case CheckInstances.IGNORE:
                        pass
                cell.insert_vinsts(recursive=False)
                if snap_ports:
                    for port in cell.to_itype().ports:
                        if port.base.dcplx_trans:
                            dup = port.base.dcplx_trans.dup()
                            dup.disp = kcl.to_um(kcl.to_dbu(port.base.dcplx_trans.disp))
                            port.dcplx_trans = dup
                if add_port_layers:
                    for port in cell.to_itype().ports:
                        if port.layer in cell.kcl.netlist_layer_mapping:
                            if port.base.trans:
                                edge = kdb.Edge(
                                    kdb.Point(0, -port.width // 2),
                                    kdb.Point(0, port.width // 2),
                                )
                                cell.shapes(
                                    cell.kcl.netlist_layer_mapping[port.layer]
                                ).insert(port.trans * edge)
                                if port.name:
                                    cell.shapes(
                                        cell.kcl.netlist_layer_mapping[port.layer]
                                    ).insert(kdb.Text(port.name, port.trans))
                            else:
                                dwidth = kcl.to_um(port.width)
                                dedge = kdb.DEdge(
                                    kdb.DPoint(0, -dwidth / 2),
                                    kdb.DPoint(0, dwidth / 2),
                                )
                                cell.shapes(
                                    cell.kcl.netlist_layer_mapping[port.layer]
                                ).insert(port.dcplx_trans * dedge)
                                if port.name:
                                    cell.shapes(
                                        cell.kcl.netlist_layer_mapping[port.layer]
                                    ).insert(
                                        kdb.DText(
                                            port.name,
                                            port.dcplx_trans.s_trans(),
                                        )
                                    )
                # post process the cell
                for pp in post_process:
                    pp(cell)  # type: ignore[arg-type]
                cell.base.lock()
                if cell.kcl != kcl:
                    raise ValueError(
                        "The KCell created must be using the same"
                        " KCLayout object as the @cell decorator. "
                        f"{kcl.name!r} != {cell.kcl.name!r}. Please make sure "
                        "to use @kcl.cell and only use @cell for cells which "
                        "are created through kfactory.kcl. To create KCells not"
                        " in the standard KCLayout, use either "
                        "custom_kcl.kcell() or KCell(kcl=custom_kcl)."
                    )
                return output_type(base=cell.base)

            with kcl.thread_lock:
                cell_ = wrapped_cell(**params)
                if cell_.destroyed():
                    # If any cell has been destroyed, we should clean up the cache.
                    # Delete all the KCell entrances in the cache which have
                    # `destroyed() == True`
                    deleted_cell_hashes: list[_HashedTuple] = [
                        _hash_item
                        for _hash_item, _cell_item in cache.items()
                        if _cell_item.destroyed()
                    ]
                    for _dch in deleted_cell_hashes:
                        del cache[_dch]
                    cell_ = wrapped_cell(**params)

                if info is not None:
                    cell_.info.update(info)

                return cell_

        self._f = wrapper_autocell
        self.cache = cache
        self.name = None
        if hasattr(f, "__name__"):
            self.name = f.__name__
        elif hasattr(f, "func"):
            self.name = f.func.__name__

    def __call__(self, *args: KCellParams.args, **kwargs: KCellParams.kwargs) -> K:
        return self._f(*args, **kwargs)
