import pytest
import kfactory as kf
from collections.abc import Callable
from tempfile import NamedTemporaryFile


def test_enclosure_name(straight_factory_dbu: Callable[..., kf.KCell]) -> None:
    wg = straight_factory_dbu(width=1000, length=10000)
    assert wg.name == "straight_W1000_L10000_LWG_EWGSTD"
    wg.show()


def test_circular_snapping(LAYER: kf.LayerEnum) -> None:
    b = kf.cells.circular.bend_circular(width=1, radius=10, layer=LAYER.WG, angle=90)
    assert b.ports["o2"].dcplx_trans.disp == kf.kcl.to_um(b.ports["o2"].trans.disp)


def test_euler_snapping(LAYER: kf.LayerEnum) -> None:
    b = kf.cells.euler.bend_euler(width=1, radius=10, layer=LAYER.WG, angle=90)
    assert b.ports["o2"].dcplx_trans.disp == kf.kcl.to_um(b.ports["o2"].trans.disp)


def test_no_snap(LAYER: kf.LayerEnum) -> None:
    c = kf.KCell()

    c.create_port(
        dwidth=1,
        dcplx_trans=kf.kdb.DCplxTrans(1, 90, False, 0.0005, 0),
        layer=LAYER.WG,
    )

    p = c.ports[0]

    assert p.dcplx_trans.disp != c.kcl.to_um(p.trans.disp)


def test_namecollision(LAYER: kf.LayerEnum) -> None:
    b1 = kf.cells.circular.bend_circular(width=1, radius=10.5, layer=LAYER.WG)
    b2 = kf.cells.circular.bend_circular(width=1, radius=10.5000005, layer=LAYER.WG)

    assert b1.name != b2.name


def test_nested_dic() -> None:
    @kf.kcl.cell
    def recursive_dict_cell(d: dict[str, dict[str, str] | str]) -> kf.KCell:
        c = kf.KCell()
        return c

    recursive_dict_cell({"test": {"test2": "test3"}, "test4": "test5"}).show()


def test_ports_cell(LAYER: kf.LayerEnum) -> None:
    c = kf.KCell()
    c.create_port(
        name="o1",
        dwidth=1,
        dcplx_trans=kf.kdb.DCplxTrans(1, 90, False, 0.0005, 0),
        layer=LAYER.WG,
    )
    assert c["o1"]
    assert "o1" in c.ports


def test_ports_instance(LAYER: kf.LayerEnum) -> None:
    c = kf.KCell()
    c.create_port(
        name="o1",
        dwidth=1,
        dcplx_trans=kf.kdb.DCplxTrans(1, 90, False, 0.0005, 0),
        layer=LAYER.WG,
    )
    c2 = kf.KCell()
    ref = c2 << c
    assert c["o1"]
    assert "o1" in c.ports
    assert ref["o1"]
    assert "o1" in ref.ports


def test_getter(LAYER: kf.LayerEnum) -> None:
    c = kf.KCell()
    c << kf.cells.straight.straight(width=1, length=10, layer=LAYER.WG)
    assert c.y == 0
    assert c.dy == 0


def test_array(straight: kf.KCell) -> None:
    c = kf.KCell()
    wg_array = c.create_inst(
        straight, a=kf.kdb.Vector(15_000, 0), b=kf.kdb.Vector(0, 3_000), na=3, nb=5
    )
    for b in range(5):
        for a in range(3):
            wg_array["o1", a, b]
            wg_array["o1", a, b]


def test_array_indexerror(straight: kf.KCell) -> None:
    c = kf.KCell()
    wg_array = c.create_inst(
        straight, a=kf.kdb.Vector(15_000, 0), b=kf.kdb.Vector(0, 3_000), na=3, nb=5
    )
    regex = kf.config.logfilter.regex
    kf.config.logfilter.regex = r"^An error has been caught in function '__getitem__'"
    with pytest.raises(IndexError):
        wg_array["o1", 3, 5]
        wg_array["o1", 3, 5]
    kf.config.logfilter.regex = regex


def test_invalid_array(monkeypatch: pytest.MonkeyPatch, straight: kf.KCell) -> None:
    c = kf.KCell()
    wg = c.create_inst(straight)
    regex = kf.config.logfilter.regex
    kf.config.logfilter.regex = r"^An error has been caught in function '__getitem__'"
    with pytest.raises(KeyError):
        for b in range(1):
            for a in range(1):
                wg["o1", a, b]
                wg["o1", a, b]
    kf.config.logfilter.regex = regex


def test_cell_decorator_error() -> None:
    kcl2 = kf.KCLayout("decorator_test")

    @kf.kcl.cell
    def wrong_cell() -> kf.KCell:
        c = kcl2.kcell("wrong_test")
        return c

    regex = kf.config.logfilter.regex
    kf.config.logfilter.regex = (
        r"^An error has been caught in function 'wrapper_autocell'"
    )
    with pytest.raises(ValueError):
        wrong_cell()
    kf.config.logfilter.regex = regex


def test_info() -> None:
    @kf.kcl.cell(info={"test": 42})
    def test_info_cell(test: int) -> kf.KCell:
        return kf.kcl.kcell()

    c = test_info_cell(42)
    assert c.info["test"] == 42


def test_flatten(LAYER: kf.LayerEnum) -> None:
    c = kf.KCell()
    _ = c << kf.cells.straight.straight(width=1, length=10, layer=LAYER.WG)
    assert len(c.insts) == 1, "c.insts should have 1 inst after adding a cell"
    c.flatten()
    assert len(c.insts) == 0, "c.insts should have 0 insts after flatten()"


def test_size_info(LAYER: kf.LayerEnum) -> None:
    c = kf.KCell()
    ref = c << kf.cells.straight.straight(width=1, length=10, layer=LAYER.WG)
    assert ref.size_info.ne[0] == 10000
    assert ref.dsize_info.ne[0] == 10


def test_overwrite() -> None:
    kcl = kf.KCLayout("CELL_OVERWRITE")

    @kcl.cell
    def test_overwrite_cell() -> kf.KCell:
        c = kcl.kcell()
        return c

    c1 = test_overwrite_cell()

    @kcl.cell(overwrite_existing=True)  # type: ignore[no-redef]
    def test_overwrite_cell() -> kf.KCell:
        c = kcl.kcell()
        return c

    c2 = test_overwrite_cell()

    assert c2 is not c1
    assert c1._destroyed()


def test_layout_cache() -> None:
    kcl_write = kf.KCLayout("TEST_LAYOUT_CACHE_WRITE")
    kcl_read = kf.KCLayout("TEST_LAYOUT_CACHE_READ")

    @kcl_write.cell(basename="straight")
    def write_straight() -> kf.KCell:
        c = kcl_write.kcell()
        c.shapes(kcl_write.layer(1, 0)).insert(kf.kdb.Box(10_000, 1000))
        return c

    s_write = write_straight()
    tf = NamedTemporaryFile(suffix=".gds.gz")
    kcl_write.write(tf.name)
    kcl_read.read(tf.name)

    @kcl_read.cell(basename="straight", layout_cache=True)
    def read_straight() -> kf.KCell:
        c = kcl_read.kcell()
        c.shapes(kcl_read.layer(1, 0)).insert(kf.kdb.Box(5000, 1000))
        return c

    s_read = read_straight()
    assert s_write.bbox() == s_read.bbox()
