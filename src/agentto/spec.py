"""Design-intent specification, schema v0.

A spec is the single source of truth for a design problem: the GUI and the agent both edit it through the same
tools, every edit is logged, and the solver problem is derived from it deterministically (problem.py). Lengths
and forces are in the spec's own units; nothing in a spec depends on the mesh.

Region coordinates are absolute, in the domain frame (origin at the domain's min corner). A face patch gives
its centre and full size in the face's two in-plane axes, in increasing axis order: (y, z) on x faces,
(x, z) on y faces, (x, y) on z faces.
"""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import SCHEMA_VERSION

Face = Literal["x-", "x+", "y-", "y+", "z-", "z+"]
Axis = Literal["x", "y", "z"]
Vec3 = tuple[float, float, float]
Vec2 = tuple[float, float]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- regions ---------------------------------------------------------------

class FaceRegion(_M):
    kind: Literal["face"] = "face"
    face: Face


class PatchRegion(_M):
    """Rectangle (or the ellipse inscribed in it, e.g. a washer face) on a domain face; nodes within half the
    size (plus a rounding tolerance) of the centre."""
    kind: Literal["patch"] = "patch"
    face: Face
    center: Vec2
    size: Vec2 = Field(description="full widths along the face's two in-plane axes")
    shape: Literal["rect", "ellipse"] = "rect"


class EdgeRegion(_M):
    """Grid line parallel to `axis`, through the point `at` in the other two axes (increasing axis order)."""
    kind: Literal["edge"] = "edge"
    axis: Axis
    at: Vec2
    span: Vec2 | None = Field(None, description="optional [start, end] along the axis; default: full length")


class BoxRegion(_M):
    kind: Literal["box"] = "box"
    min: Vec3
    max: Vec3

    @model_validator(mode="after")
    def _ordered(self):
        if any(a >= b for a, b in zip(self.min, self.max)):
            raise ValueError("box min must be < max on every axis")
        return self


class CylinderRegion(_M):
    """Solid circular cylinder: axis direction, a point on the axis, radius and [start, end] along the axis."""
    kind: Literal["cylinder"] = "cylinder"
    axis: Axis
    center: Vec2 = Field(description="axis position in the other two axes (increasing axis order)")
    radius: float = Field(gt=0)
    span: Vec2

    @model_validator(mode="after")
    def _ordered(self):
        if self.span[0] >= self.span[1]:
            raise ValueError("cylinder span start must be < end")
        return self


NodeRegion = Annotated[Union[FaceRegion, PatchRegion, EdgeRegion, BoxRegion, CylinderRegion],
                       Field(discriminator="kind")]
VolumeRegion = Annotated[Union[BoxRegion, CylinderRegion], Field(discriminator="kind")]


# ---- spec parts ------------------------------------------------------------

class Units(_M):
    length: Literal["mm", "m", "in"] = "mm"
    force: Literal["N", "kN", "lbf"] = "N"


class BoxDomain(_M):
    kind: Literal["box"] = "box"
    size: Vec3

    @model_validator(mode="after")
    def _positive(self):
        if any(s <= 0 for s in self.size):
            raise ValueError("domain size must be positive")
        return self


class Material(_M):
    name: str = "steel"
    E: float = Field(210e3, gt=0, description="Young's modulus in force/length^2 of the spec units")
    nu: float = Field(0.3, gt=-1.0, lt=0.5)


class Support(_M):
    id: str
    region: NodeRegion
    fix: list[Axis] = Field(default_factory=lambda: ["x", "y", "z"], min_length=1)
    note: str = ""


class Load(_M):
    """Total force on the region, shared equally by its nodes (mesh-independent total)."""
    id: str
    region: NodeRegion
    force: Vec3
    distribution: Literal["uniform_nodes"] = "uniform_nodes"
    note: str = ""


class Passive(_M):
    id: str
    kind: Literal["solid", "void"]
    region: VolumeRegion
    note: str = ""


class Resolution(_M):
    """Mesh size: either `h` (element edge, length units) or `cells_short_edge` (elements on the shortest side)."""
    h: float | None = Field(None, gt=0)
    cells_short_edge: int | None = Field(None, ge=2)

    @model_validator(mode="after")
    def _one(self):
        if (self.h is None) == (self.cells_short_edge is None):
            raise ValueError("give exactly one of h or cells_short_edge")
        return self


class Spec(_M):
    schema_version: Literal["agentto.spec/v0"] = SCHEMA_VERSION
    name: str
    description: str = ""
    units: Units = Units()
    domain: BoxDomain
    material: Material = Material()
    volume_fraction: float = Field(gt=0, lt=1, description="fraction of the design (non-passive) volume")
    supports: list[Support] = Field(min_length=1)
    loads: list[Load] = Field(min_length=1)
    passive: list[Passive] = Field(default_factory=list)
    resolution: Resolution
    min_feature: float | None = Field(None, gt=0, description="smallest member the user needs, length units")
    meta: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_ids(self):
        ids = [x.id for x in (*self.supports, *self.loads, *self.passive)]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"duplicate ids: {dup}")
        return self


def json_schema() -> dict:
    return Spec.model_json_schema()
