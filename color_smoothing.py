import numpy as np
import math
from constants import *

def get_neighbouring_segments(segments, lum, limit):
    number_of_segments = np.max(segments) + 1
    neighbours = [set() for _ in range(number_of_segments)]
    for (x, y), value in np.ndenumerate(segments):
        if y < len(segments[x]) - 1:
            adj_value = segments[x][y + 1]
            if adj_value != value and np.linalg.norm(lum[value] - lum[adj_value]) < limit:
                neighbours[value].add(adj_value)
                neighbours[adj_value].add(value)

        if x < len(segments) - 1:
            adj_value = segments[x + 1][y]
            if adj_value != value and np.linalg.norm(lum[value] - lum[adj_value]) < limit:
                neighbours[value].add(adj_value)
                neighbours[adj_value].add(value)

    return neighbours

def smooth_colors(predicted_u, predicted_v, segments, lum, limit):
    smoothed_u = np.copy(predicted_u)
    smoothed_v = np.copy(predicted_v)

    neighbours = get_neighbouring_segments(segments, lum, limit)
    number_of_segments = predicted_v.shape[0]
    for _ in range(ICM_ITERATIONS):
        minimized_u = np.zeros(number_of_segments)
        minimized_v = np.zeros(number_of_segments)

        for segment_idx in range(number_of_segments):
            u_energy = math.inf
            v_energy = math.inf
            u_best = None
            v_best = None

            for u in np.arange(U_MIN, U_MAX, COLOR_INCREMENT):
                cost_assumption_1 = ((u - predicted_u[segment_idx]) ** 2)
                cost_assumption_2 = NEIGHBOUR_IMPORTANCE * sum([(u - smoothed_u[neighbour]) ** 2
                                                                for neighbour in neighbours[segment_idx]])
                u_candidate_energy = cost_assumption_1 + cost_assumption_2
                if u_candidate_energy < u_energy:
                    u_energy = u_candidate_energy
                    u_best = u
            minimized_u[segment_idx] = u_best

            for v in np.arange(V_MIN, V_MAX, COLOR_INCREMENT):
                cost_assumption_1 = ((v - predicted_v[segment_idx]) ** 2)
                cost_assumption_2 = NEIGHBOUR_IMPORTANCE * sum([(v - smoothed_v[neighbour]) ** 2
                                                                for neighbour in neighbours[segment_idx]])
                v_candidate_energy = cost_assumption_1 + cost_assumption_2
                if v_candidate_energy < v_energy:
                    v_energy = v_candidate_energy
                    v_best = v
            minimized_v[segment_idx] = v_best

        u_delta = np.linalg.norm(smoothed_u - minimized_u)
        v_delta = np.linalg.norm(smoothed_v - minimized_v)

        smoothed_u = minimized_u
        smoothed_v = minimized_v
        if u_delta < ICM_BREAK_CONDITION and v_delta < ICM_BREAK_CONDITION:
            break

    return smoothed_u, smoothed_v