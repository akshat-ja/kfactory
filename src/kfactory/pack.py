import hashlib
from typing import Any, List, Optional, Tuple, Union

import kfactory as kf
import numpy as np
import rectpack


@kf.cell
def bin(bin_hash: str) -> kf.KCell:
    """
    Implementation cell for get_bin. Don't call it directly.
    """
    c = kf.KCell()

    for cell, xy in zip(bin.cells, bin.positions):
        inst = c << cell
        inst.move(xy)

    return c


def get_bin(cells: List[kf.KCell], positions: List[Tuple[int, int]]) -> kf.KCell:
    """
    Instantiates cells at different displacements.

    Args:
        cells: list of KCells
        positions: list of (x,y) coordinates for the origins of the cells
    Returns:
        KCell
    """
    arg_hash = hashlib.sha256()

    for cell in cells:
        arg_hash.update(cell.name.encode())
    for pos in positions:
        arg_hash.update(str(pos).encode())

    bin.cells = cells
    bin.positions = positions

    return bin(arg_hash.hexdigest()[:6])


def pack_bins(
    kcells: List[kf.KCell],
    bins: List[Tuple[int, int]],
    bbox_enlarge: int = 0,
    bbox_layer: Optional[Union[kf.LayerEnum, int]] = None,
    allow_overflow=False,
) -> List[kf.KCell]:
    """
    Attempt to pack a list of kcells into the specified rectangular bins.
    By default, an exception will be raised if not all cells can be packed.

    Args:
        kcells: list of KCells to pack
        bins: list of bin sizes, tuples (width, height) in DBU
        bbox_enlarge: amount to enlarge all bounding boxes by
        bbox_layer: if provided, use only this layer to calculate bounding boxes
        allow_overflow: if True, will not raise an exception when not all rects are packed
    Returns:
        list of KCells representing non-empty bins. Might not use all provided bins!

    Example:
        cells = [
            bend_coax(180),
            bend_coax(90),
        ]
        many_cells = random.choices(cells, k=101)
        bins = pack.pack_bins(many_cells, [(mm(1), mm(2)), (mm(2),mm(3))])
    """
    # ==== Get bounding boxes

    def _size(bbox: kf.kdb.Box):
        return (bbox.width(), bbox.height())

    if bbox_layer is None:
        bboxes = [kcell.bbox().enlarge(bbox_enlarge) for kcell in kcells]
    else:
        bboxes = [kcell.bbox(bbox_layer).enlarge(bbox_enlarge) for kcell in kcells]

    rects = [(bbox.width(), bbox.height()) for bbox in bboxes]
    origins = [(bbox.left, bbox.bottom) for bbox in bboxes]

    # ==== Pack

    packer = rectpack.newPacker(rotation=False)

    for idx, r in enumerate(rects):
        packer.add_rect(*r, idx)
    for b in bins:
        packer.add_bin(*b)

    packer.pack()

    # ==== Instantiate bin cells

    out_bin_cells = []

    num_packed_cells = 0

    for packer_bin in packer:
        bin_cells = []
        bin_xy = []

        for rect in packer_bin:
            cell = kcells[rect.rid]
            cell_rect = rects[rect.rid]
            cell_origin = origins[rect.rid]
            xy = (rect.x - cell_origin[0], rect.y - cell_origin[1])
            bin_cells.append(cell)
            bin_xy.append(xy)

        bin_cell = get_bin(bin_cells, bin_xy)
        out_bin_cells.append(bin_cell)
        num_packed_cells += len(bin_cells)

    if num_packed_cells != len(kcells):
        if allow_overflow:
            print(f"WARNING: Could only pack {num_packed_cells} out of {len(kcells)} cells.")
        else: # if not allow_overflow:
            raise RuntimeError(f"ERROR: Could only pack {num_packed_cells} out of {len(kcells)} cells.")
    else:
        print(f"COMPLETED: {num_packed_cells} out of {len(kcells)} cells packed.")

    return out_bin_cells
