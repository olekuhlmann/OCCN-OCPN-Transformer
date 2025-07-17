"""
    PM4Py – A Process Mining Library for Python
Copyright (C) 2024 Process Intelligence Solutions UG (haftungsbeschränkt)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see this software project's root or
visit <https://www.gnu.org/licenses/>.

Website: https://processintelligence.solutions
Contact: info@processintelligence.solutions
"""

import random
from pm4py.algo.simulation.playout.ocpn.variants.utils import feasible_traces_to_ocel
from pm4py.objects.ocpn.semantics import OCPetriNetSemantics
from pm4py.objects.ocel.obj import OCEL
from pm4py.objects.ocpn.obj import OCPetriNet, OCMarking
from pm4py.objects.ocel import constants
from pm4py.util import exec_utils
from enum import Enum
from typing import Optional, Dict, Any, Set, Union


class Parameters(Enum):
    EVENT_ID = constants.PARAM_EVENT_ID
    EVENT_ACTIVITY = constants.PARAM_EVENT_ACTIVITY
    EVENT_TIMESTAMP = constants.PARAM_EVENT_TIMESTAMP
    OBJECT_ID = constants.PARAM_OBJECT_ID
    OBJECT_TYPE = constants.PARAM_OBJECT_TYPE
    MAX_BINDINGS_PER_ACTIVITY = "maxBindingsPerActivity"
    NUM_TRACES = "num_traces"
    RETURN_TRACES = "return_traces"
    OCPETRINET_SEMANTICS = "ocpetrinet_semantics"




def apply(
    net: OCPetriNet,
    initial_marking: OCMarking,
    final_marking: OCMarking,
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> OCEL:
    """
    Compute playout of an object-centric Petri net generating an OCEL (random walk;
    any activity may only be executed a limited number of times as specified).
    Note that this method returns a subset of all possible traces without duplicates,
    and that this subset is not a random sample of all possible traces.
    Also note that this method is very inefficient if the requested number of
    traces is close to or larger than the number of traces in the language of the model.

    Parameters
    -----------
    net
        Object-centric Petri net to play-out
    initial_marking
        Initial marking of the object-centric Petri net
    final_marking
        Final marking of the object-centric Petri net
    parameters
        Parameters of the algorithm:
            Parameters.NUM_TRACES -> Number of traces to generate. Not optional.
            Parameters.MAX_BINDINGS_PER_ACTIVITY -> Maximum bindings per activity
            Parameters.RETURN_TRACES -> If True, return traces instead of OCEL
            Parameters.OCPETRINET_SEMANTICS -> Object-centric Petri net semantics
    """
    if parameters is None or Parameters.NUM_TRACES.value not in parameters.keys():
        raise ValueError(f"Parameter {Parameters.NUM_TRACES} is required")

    num_traces = exec_utils.get_param_value(Parameters.NUM_TRACES, parameters, 0)
    return_traces = exec_utils.get_param_value(
        Parameters.RETURN_TRACES, parameters, False
    )
    max_bindings_per_activity = exec_utils.get_param_value(
        Parameters.MAX_BINDINGS_PER_ACTIVITY, parameters, 3
    )
    semantics = exec_utils.get_param_value(
        Parameters.OCPETRINET_SEMANTICS,
        parameters,
        OCPetriNetSemantics(),
    )

    # Save transitions as ids for memory efficiency; create lookup table for conversion
    all_transitions = sorted(list(net.transitions), key=lambda t: t.name)
    transition_to_idx = {t: i for i, t in enumerate(all_transitions)}

    # State: tuple[OCMarking, transition_counts, trace]
    # trace is a tuple of events: (transition_index, frozenset of object_ids)
    # index in transition_counts corresponds to index in transition_to_idx
    initial_state = (initial_marking, (0,) * len(all_transitions), [])

    # final set of traces
    feasible_traces = set()

    for _ in range(num_traces):
        success = _random_walk(
            net,
            initial_state,
            final_marking,
            max_bindings_per_activity,
            semantics,
            transition_to_idx,
            feasible_traces,
        )
        if not success:
            # num_traces is larger than the number of feasible traces
            break

    if return_traces:
        # Inverse the transition_to_idx mapping to get transition labels
        idx_to_transition = {v: k for k, v in transition_to_idx.items()}
        return (list(feasible_traces), idx_to_transition)
    else:
        return feasible_traces_to_ocel(
            iter(feasible_traces), initial_marking, all_transitions, parameters
        )


def _random_walk(
    net: OCPetriNet,
    state: tuple,
    final_marking: OCMarking,
    max_bindings_per_activity: int,
    semantics: OCPetriNetSemantics,
    t_to_idx: dict,
    feasible_traces: Set,
) -> bool:
    """
    Perform a random walk through the object-centric Petri net.
    Whenever multiple transitions are enabled, one is chosen at random.
    Whenever a transition has multiple available bindings, one is chosen at random.
    If the random walk reaches a deadlock, it backtracks and chooses a different path.
    If the random walk reaches the final marking, it adds the trace to the set of feasible traces.
    If the trace is already in the set of feasible traces,
    it does not add the trace again but backtracks to find a different path.

    Parameters
    -----------
    net
        Object-centric Petri net to play-out
    state
        Current state of the random walk (marking, transition counts, trace)
    final_marking
        Final marking of the object-centric Petri net
    max_bindings_per_activity
        Maximum bindings per activity
    semantics
        Object-centric Petri net semantics
    t_to_idx
        A lookup dictionary mapping transition objects to their integer indices.
    feasible_traces
        Set of feasible traces to avoid duplicates

    Returns
    -----------
    bool
        True if a new trace was generated, False otherwise.
    """
    marking, transition_counts, trace = state

    if marking == final_marking:
        # check if the trace is already in the set of feasible traces
        if trace not in feasible_traces:
            feasible_traces.add(trace)
            return True
        return False

    # explore next state
    enabled_transitions = list(semantics.enabled_transitions(net, marking))
    # sort them randomly to ensure a random walk
    random.shuffle(enabled_transitions)

    for t in enabled_transitions:
        # Check if the transition can be fired
        transition_idx = t_to_idx[t]
        if transition_counts[transition_idx] >= max_bindings_per_activity:
            continue

        # Create new transition counts
        new_transition_counts = list(transition_counts)
        new_transition_counts[transition_idx] += 1
        new_counts_tuple = tuple(new_transition_counts)

        # Get all possible bindings for the transition
        bindings = list(semantics.get_possible_bindings(net, t, marking))
        # sort them randomly to ensure a random walk
        random.shuffle(bindings)

        for binding in bindings:
            new_marking = semantics.fire(net, t, marking, binding)
            object_ids = [oi for objs in binding.values() for oi in objs]
            new_trace = list(trace) + [(transition_idx, frozenset(object_ids))]
            new_trace_tuple = tuple(new_trace)

            new_state = (new_marking, new_counts_tuple, new_trace_tuple)

            # Recursively explore the next state
            if _random_walk(
                net,
                new_state,
                final_marking,
                max_bindings_per_activity,
                semantics,
                t_to_idx,
                feasible_traces,
            ):
                return True

            # backtrack with next loop iteration otherwise

    return False
