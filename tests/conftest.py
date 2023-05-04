import pytest
import kfactory as kf
from functools import partial


class LAYER_CLASS(kf.LayerEnum):
    WG = (1, 0)
    WGCLAD = (111, 0)


@pytest.fixture
def LAYER():
    return LAYER_CLASS


@pytest.fixture
def wg_enc(LAYER):
    return kf.utils.Enclosure(name="WGSTD", sections=[(LAYER.WGCLAD, 0, 2000)])


@pytest.fixture
def waveguide_factory(LAYER, wg_enc):
    return partial(kf.cells.dbu.waveguide, layer=LAYER.WG, enclosure=wg_enc)


@pytest.fixture
def bend90(LAYER, wg_enc):
    return kf.cells.circular.bend_circular(
        width=1, radius=10, layer=LAYER.WG, enclosure=wg_enc, theta=90
    )


@pytest.fixture
def bend180(LAYER, wg_enc):
    return kf.cells.circular.bend_circular(
        width=1, radius=10, layer=LAYER.WG, enclosure=wg_enc, theta=180
    )


@pytest.fixture
def bend90_euler(LAYER, wg_enc):
    return kf.cells.euler.bend_euler(
        width=1, radius=10, layer=LAYER.WG, enclosure=wg_enc, theta=90
    )


@pytest.fixture
def bend180_euler(LAYER, wg_enc):
    return kf.cells.euler.bend_euler(
        width=1, radius=10, layer=LAYER.WG, enclosure=wg_enc, theta=180
    )


@pytest.fixture
def optical_port(LAYER):
    return kf.Port(
        name="o1",
        trans=kf.kdb.Trans.R0,
        layer=LAYER.WG,
        width=1000,
        port_type="optical",
    )


@pytest.fixture
def cells():
    return [
        kf.cells.bezier,
        kf.cells.euler,
        kf.cells.circular,
        kf.cells.taper,
        kf.cells.waveguide,
    ]


@pytest.fixture
def pdk(LAYER, waveguide_factory, wg_enc):
    pdk = kf.pdk.Pdk(layers=LAYER, name="TEST_PDK", cell_factories={"wg": waveguide_factory, "bend": kf.cells.circular.bend_circular, "bend_euler": kf.cells.euler.bend_euler, "taper": kf.cells.taper.taper, "bezier": kf.cells.bezier.bend_s})
    pdk.register_cells(waveguide=waveguide_factory)
    pdk.register_enclosures(wg=wg_enc)
    pdk.activate()
    return pdk


# @pytest.fixture
# def wg():
#     return LAYER.WG
