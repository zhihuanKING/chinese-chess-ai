// pybind11 bindings for the Xiangqi kernel (INTERFACES.md §3).
// Exposes Position and the result/side constants under module `_xqcore`
// (imported as `xqai._xqcore`).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "xq/position.h"
#include "xq/search.h"

namespace py = pybind11;
using namespace xq;

PYBIND11_MODULE(_xqcore, m) {
    m.doc() = "Xiangqi rules kernel + Alpha-Beta engine (xqai._xqcore)";

    // Side / result constants.
    m.attr("RED") = static_cast<int>(RED);
    m.attr("BLACK") = static_cast<int>(BLACK);
    m.attr("ONGOING") = static_cast<int>(ONGOING);
    m.attr("RED_WIN") = static_cast<int>(RED_WIN);
    m.attr("BLACK_WIN") = static_cast<int>(BLACK_WIN);
    m.attr("DRAW") = static_cast<int>(DRAW);
    m.attr("ACTION_DIM") = kActionDim;

    py::class_<Position>(m, "Position")
        .def(py::init<>())
        .def_static("from_fen", &Position::from_fen, py::arg("fen"))
        .def("fen", &Position::fen)
        .def("side_to_move", [](const Position& p) { return static_cast<int>(p.side_to_move()); })
        .def("legal_moves", &Position::legal_moves)
        .def("is_check", &Position::is_check)
        .def("push", &Position::push, py::arg("move"))
        .def("pop", &Position::pop)
        .def("result", [](const Position& p) { return static_cast<int>(p.result()); })
        .def("zobrist", &Position::zobrist)
        .def("board",
             [](const Position& p) {
                 // length-90 int8 view as Python bytes (row-major, §1 encoding).
                 return py::bytes(reinterpret_cast<const char*>(p.board_data()),
                                  kSquares);
             })
        .def("repetition_count", &Position::repetition_count)
        .def("ply", &Position::ply)
        .def_static("perft", &Position::perft, py::arg("fen"), py::arg("depth"))
        // Engine convenience: search best move with a time budget (ms).
        .def("search",
             [](Position& p, int time_ms) { return search(p, time_ms); },
             py::arg("time_ms") = 1000)
        .def("__repr__",
             [](const Position& p) { return "<Position fen='" + p.fen() + "'>"; });

    // Module-level engine entry too, matching INTERFACES.md §8 signature.
    m.def("search",
          [](Position& p, int time_ms) { return search(p, time_ms); },
          py::arg("position"), py::arg("time_ms") = 1000,
          "Alpha-Beta search: returns best move integer (from*90+to).");
    m.def("evaluate", &evaluate, py::arg("position"),
          "Static evaluation in centipawns (side-to-move perspective).");
}
