import dolfinx_adjoint.graph as graph
from dolfinx_adjoint.edge import Edge
from dolfinx_adjoint.node import AbstractNode


def test_get_path_marks_only_edges_on_path():
    g = graph.Graph()

    start = AbstractNode(object(), name="start")
    mid = AbstractNode(object(), name="mid")
    end = AbstractNode(object(), name="end")
    other_a = AbstractNode(object(), name="other_a")
    other_b = AbstractNode(object(), name="other_b")

    for node in [start, mid, end, other_a, other_b]:
        g.add_node(node)

    # Path in reverse graph: start -> mid -> end
    # Edge direction in this codebase is predecessor -> successor,
    # and the reverse graph uses successor -> predecessor.
    e1 = Edge(mid, start)
    e2 = Edge(end, mid)
    e3 = Edge(other_b, other_a)

    g.add_edge(e1)
    g.add_edge(e2)
    g.add_edge(e3)

    g.get_path(id(start), id(end))

    assert e1.marked is True
    assert e2.marked is True
    assert e3.marked is False


def test_get_path_with_no_connection_marks_none():
    g = graph.Graph()

    start = AbstractNode(object(), name="start")
    mid = AbstractNode(object(), name="mid")
    end = AbstractNode(object(), name="end")
    other_a = AbstractNode(object(), name="other_a")
    other_b = AbstractNode(object(), name="other_b")

    for node in [start, mid, end, other_a, other_b]:
        g.add_node(node)

    e1 = Edge(mid, start)
    e2 = Edge(end, mid)
    e3 = Edge(other_b, other_a)

    g.add_edge(e1)
    g.add_edge(e2)
    g.add_edge(e3)

    g.get_path(id(other_a), id(end))

    assert e1.marked is False
    assert e2.marked is False
    assert e3.marked is False
