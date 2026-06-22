import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import numpy as np

from ..gmsh_dep import GmshContext
from ..helper.index_mapping import IndexMapping
# from .boundary import BoundarySpec
# from .edges import edges_of_all_elements, unique_edges


@dataclass(frozen=True)
class PhysicalGroupBase(ABC):
    """Base class for physical groups, which are named collections of gmsh entities.
    PhysicalGroup classes are expected to persist after closing the gmsh context, but are
    permitted to use them during construction.
    """

    name: str

    @abstractmethod
    def remap_indices(
        self,
        /,
        node_mapping: IndexMapping | None = None,
        element_mapping: IndexMapping | None = None,
    ) -> "PhysicalGroupBase":
        """Remaps the node and element indices of this physical group. Indices are remapped
        according to <node/element>_mapping.apply(indices)

        Args:
            node_mapping (IndexMapping | None, optional): the index mapping that takes the
                node indices to their new values, or None if the node indices should be kept
                the same. Defaults to None.
            element_mapping (IndexMapping | None, optional): the index mapping that takes the
                element indices to their new values, or None if the element indices should be kept
                the same. Defaults to None.

        Returns:
            PhysicalGroupBase: The physical group with remapped values
        """
        raise NotImplementedError

    def remap_nodes(self, node_mapping: IndexMapping) -> "PhysicalGroupBase":
        """If a node reindexing occurs, call this method to apply it to the physical group.
        Nodes are remapped according to node_mapping.apply(node_indices)

        Args:
            node_mapping (IndexMapping):
        """
        return self.remap_indices(node_mapping=node_mapping)

    def remap_elements(self, element_mapping: IndexMapping) -> "PhysicalGroupBase":
        """If an element reindexing occurs, call this method to apply it to the physical group.
        Elements are remapped according to element_mapping.apply(element_indices)

        Args:
            element_mapping (IndexMapping):
        """
        return self.remap_indices(element_mapping=element_mapping)

    @abstractmethod
    def get_all_points(self, include_from_higher_dim: bool = False) -> np.ndarray:
        """Recovers all of the points (node indices) from this physical group.
        When `include_from_higher_dim` is set to true, nodes from edges and elements
        are included. That is, higher dimensional physical groups are not skipped.

        Args:
            include_from_higher_dim (bool, optional): whether to pull from higher
                dimensional groups. Defaults to False.

        Returns:
            np.ndarray: a shape (N,) array containing node indices.
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_edges(
        self, include_from_higher_dim: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recovers all of the edges (element index + EdgeType) from this physical group.
        When `include_from_higher_dim` is set to true, nodes from elements
        are included. That is, higher dimensional physical groups are not skipped.

        Args:
            include_from_higher_dim (bool, optional): whether to pull from higher
                dimensional groups. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray]: Two shape (N,) integer arrays. The first stores the
                element indices, while the second stores the edge types.
        """
        raise NotImplementedError


@dataclass(frozen=True, init=False)
class SingleDimensionPhysicalGroup(PhysicalGroupBase):
    """The bulk of physical groups are stored in these, which perform the main logic of
    extracting gmsh physical groups.

    A SingleDimensionPhysicalGroup will store elements of only one number of dimensions.
    """

    dim: int

    def __init__(
        self,
        gmsh: GmshContext,
        element_nodes: np.ndarray,
        node_mapping: IndexMapping,
        node_coords: np.ndarray,
        name: str | None = None,
        tag: int | None = None,
    ):
        """Recovers the elements of a given dimension associated with a physical group tag or name.
        the gmsh context is only used in generation.
        `element_nodes` are the un-remapped node indices of each element,
        in the same order as `element_mapping.original_tag_list`.

        Args:
            gmsh (GmshContext): gmsh context
            element_nodes (np.ndarray): the node tags (shape = (N,9)) of all elements.
            node_mapping (IndexMapping): the index mapping of node tags.
            node_coords (np.ndarray): the coordinate array, re-indexed by node_mapping.
            name (str | None, optional): identifier of the group. If multiple groups have the same
                name, then all of them with the correct dimension are taken. This behavior overrides
                the `tag` argument. If specified, `tag` is not needed.
            tag (int | None, optional): identifier of the group. If specified, `name` is not needed.
                If both `name` and `tag` are given, all tags with the given name are chosen, but if
                `tag` is not among them, an error is thrown.
        """
        if not hasattr(self.__class__, "_dim"):
            e = TypeError(
                "SingleDimensionPhysicalGroup requires subclasses to have a `_dim` attribute "
                f"to recover the dimension, but {self.__class__} does not have one."
            )
            raise e
        dim = self.__class__._dim  # pyright: ignore
        object.__setattr__(self, "dim", dim)

        # a name may correspond with multiple tags
        tags = []

        if name is None:
            if tag is None:
                e = ValueError("`name` or `tag` must be specified!")
                raise e
            name = gmsh.model.get_physical_name(dim, tag)
        else:
            for _dim, _tag in gmsh.model.get_physical_groups():
                if _dim == dim and gmsh.model.get_physical_name(_dim, _tag) == name:
                    # found a tag with this identity
                    tags.append(_tag)
        object.__setattr__(self, "name", name)

        if tag is not None and tag not in tags:
            e = ValueError(
                f"`name` {name} was specified, but given `tag` {tag} (name = "
                f"{gmsh.model.get_physical_name(dim, tag)}) was not included!"
            )
            raise e

        # we have our list of tags to pull from; dispose of them after getting all data from gmsh.

        entities = []
        for tag in tags:
            entities.extend(gmsh.model.get_entities_for_physical_group(dim, tag))

        self._populate(gmsh, element_nodes, node_mapping, node_coords, dim, entities)

    @abstractmethod
    def _populate(
        self,
        gmsh: GmshContext,
        element_nodes: np.ndarray,
        node_mapping: IndexMapping,
        node_coords: np.ndarray,
        dim: int,
        entity_tags: list[int],
    ) -> None:
        raise NotImplementedError


@dataclass(frozen=True, init=False)
class PhysicalGroup0D(SingleDimensionPhysicalGroup):
    _dim = 0
    nodes: np.ndarray

    @override
    def _populate(
        self,
        gmsh: GmshContext,
        element_nodes: np.ndarray,
        node_mapping: IndexMapping,
        node_coords: np.ndarray,
        dim: int,
        entity_tags: list[int],
    ):
        assert dim == 0
        nodelists = []
        for entity in entity_tags:
            gmsh.for_element_types_in_entity(
                dim,
                entity,
                mapping={
                    15: lambda elem_tags, node_tags: nodelists.append(
                        node_mapping.apply(node_tags)
                    )
                },
            )
        if nodelists:
            nodes = np.unique(np.concatenate(nodelists))
        else:
            nodes = np.empty((0,), dtype=np.uint64)
        object.__setattr__(self, "nodes", nodes)

    @override
    def remap_indices(
        self,
        /,
        node_mapping: IndexMapping | None = None,
        element_mapping: IndexMapping | None = None,
    ) -> "PhysicalGroupBase":
        if node_mapping is None:
            return self
        cpy = copy.copy(self)
        object.__setattr__(cpy, "nodes", node_mapping.apply(self.nodes))
        return cpy

    @override
    def get_all_points(self, include_from_higher_dim: bool = False) -> np.ndarray:
        return self.nodes

    @override
    def get_all_edges(
        self, include_from_higher_dim: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=np.uint8)


@dataclass(frozen=True, init=False)
class PhysicalGroup1D(SingleDimensionPhysicalGroup):
    _dim = 1
    edges: BoundarySpec

    # consider moving this somewhere else. The information is redundant, and is only stored to keep
    # Model and PhysicalGroup in a one-way relationship.
    nodes: np.ndarray

    @override
    def _populate(
        self,
        gmsh: GmshContext,
        element_nodes: np.ndarray,
        node_mapping: IndexMapping,
        node_coords: np.ndarray,
        dim: int,
        entity_tags: list[int],
    ):
        assert dim == 1
        object.__setattr__(
            self,
            "edges",
            BoundarySpec.from_model_entity(
                gmsh, entity_tags, element_nodes, node_mapping, node_coords
            ),
        )
        object.__setattr__(
            self,
            "nodes",
            edges_of_all_elements(element_nodes)[
                self.edges.element_inds, self.edges.element_edges
            ],
        )

    @override
    def remap_indices(
        self,
        /,
        node_mapping: IndexMapping | None = None,
        element_mapping: IndexMapping | None = None,
    ) -> "PhysicalGroupBase":
        if node_mapping is None and element_mapping is None:
            return self

        cpy = copy.copy(self)
        if element_mapping is not None:
            object.__setattr__(
                cpy, "edges", self.edges.remapped_elements(element_mapping)
            )
        if node_mapping is not None:
            object.__setattr__(cpy, "nodes", node_mapping.apply(self.nodes))
        return cpy

    @override
    def get_all_points(self, include_from_higher_dim: bool = False) -> np.ndarray:
        if include_from_higher_dim:
            return np.unique(self.nodes)
        return np.empty((0,), dtype=np.uint64)

    @override
    def get_all_edges(
        self, include_from_higher_dim: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        return unique_edges(self.edges.element_inds, self.edges.element_edges)


@dataclass(frozen=True, init=False)
class NullPhysicalGroup(PhysicalGroupBase):
    @override
    def remap_indices(
        self,
        /,
        node_mapping: IndexMapping | None = None,
        element_mapping: IndexMapping | None = None,
    ) -> "PhysicalGroupBase":
        return self

    @override
    def get_all_points(self, include_from_higher_dim: bool = False) -> np.ndarray:
        return np.empty((0,), dtype=np.uint64)

    @override
    def get_all_edges(
        self, include_from_higher_dim: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=np.uint8)


@dataclass(frozen=True, init=False)
class UnionPhysicalGroup(PhysicalGroupBase):
    physical_groups: list[PhysicalGroupBase]

    def __init__(
        self, name: str, *groups: PhysicalGroupBase, verify_names: bool = True
    ):
        super().__init__(name=name)
        object.__setattr__(self, "physical_groups", [])

        def add_group(group: PhysicalGroupBase):
            if verify_names and group.name != name:
                e = ValueError(
                    "When constructing UnionPhysicalGroup of name "
                    f"{name}: encountered different name {group.name}."
                )
                raise e

            # expand unions (basedbasedbasedbased). We should have a depth of one
            if isinstance(group, UnionPhysicalGroup):
                for subgp in group.physical_groups:
                    add_group(subgp)
                return

            # blank. We can skip
            if isinstance(group, NullPhysicalGroup):
                return

            self.physical_groups.append(group)

        for group in groups:
            add_group(group)

    @override
    def remap_indices(
        self,
        /,
        node_mapping: IndexMapping | None = None,
        element_mapping: IndexMapping | None = None,
    ) -> "PhysicalGroupBase":
        return UnionPhysicalGroup(
            self.name,
            *(
                gp.remap_indices(
                    node_mapping=node_mapping, element_mapping=element_mapping
                )
                for gp in self.physical_groups
            ),
            verify_names=False,
        )

    @override
    def get_all_points(self, include_from_higher_dim: bool = False) -> np.ndarray:
        collected = [
            gp.get_all_points(include_from_higher_dim=include_from_higher_dim)
            for gp in self.physical_groups
        ]
        if collected:
            return np.unique(np.concatenate(collected))
        else:
            return np.empty((0,), dtype=np.uint64)

    @override
    def get_all_edges(
        self, include_from_higher_dim: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        collected_elems = []
        collected_edgetypes = []
        for gp in self.physical_groups:
            elem, edge = gp.get_all_edges(
                include_from_higher_dim=include_from_higher_dim
            )
            collected_elems.append(elem)
            collected_edgetypes.append(edge)

        if collected_elems:
            return unique_edges(
                np.concatenate(collected_elems), np.concatenate(collected_edgetypes)
            )
        else:
            return np.empty((0,), dtype=np.uint64), np.empty((0,), dtype=np.uint8)


PhysicalGroup = PhysicalGroupBase


def physical_group_from_name(
    gmsh: GmshContext,
    element_nodes: np.ndarray,
    node_mapping: IndexMapping,
    node_coords: np.ndarray,
    name: str,
):
    """Recovers the elements of a given dimension associated with a physical group tag or name.
    the gmsh context is only used in generation.
    `element_nodes` are the un-remapped node indices of each element,
    in the same order as `element_mapping.original_tag_list`.

    Args:
        gmsh (GmshContext): gmsh context
        element_nodes (np.ndarray): the node tags (shape = (N,9)) of all elements.
        node_mapping (IndexMapping): the index mapping of node tags.
        node_coords (np.ndarray): the coordinate array, re-indexed by node_mapping.
        name (str): identifier of the group.
    """

    # get one group for each dimension
    pg0d = PhysicalGroup0D(gmsh, element_nodes, node_mapping, node_coords, name=name)
    pg1d = PhysicalGroup1D(gmsh, element_nodes, node_mapping, node_coords, name=name)

    if len(pg1d.edges.edge_tags) > 0:
        return pg1d
    elif len(pg0d.nodes) > 0:
        return pg0d

    return NullPhysicalGroup(name)
