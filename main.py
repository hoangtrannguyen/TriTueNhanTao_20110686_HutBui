import asyncio
import sys
import math
import random
import heapq
from collections import deque

GRID_SIZE = 4
CELL_COUNT = GRID_SIZE * GRID_SIZE
MOVES = [(-1, 0, "Lên"), (1, 0, "Xuống"), (0, -1, "Trái"), (0, 1, "Phải")]
ALGORITHMS = [
    "BFS", "DFS", "IDS", "UCS", "Manhattan Greedy", "A*", "IDA*", "Misplaced Greedy",
    "Leo đồi ngẫu nhiên", "Leo đồi đơn giản", "Simulated Annealing", "Local Beam",
    "AND-OR Search", "Online Search", "TK không biết 1 phần", "TK không thể quan sát",
    "Backtracking", "Forward checking", "AC-3", "Min-conflict",
    "Minimax", "Alpha-beta", "Expectimax", "NegaMax"
]


def cell_index(pos):
    return pos[0] * GRID_SIZE + pos[1]


def cell_bit(pos):
    return 1 << cell_index(pos)


def mask_from_cells(cells):
    mask = 0
    for pos in cells:
        mask |= cell_bit(pos)
    return mask


def dirt_cells(mask):
    result = []
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            pos = (r, c)
            if mask & cell_bit(pos):
                result.append(pos)
    return result


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def create_problem(agent, dirt):
    return {"start_state": (agent, mask_from_cells(dirt))}


def actions(state):
    pos, mask = state
    r, c = pos
    result = []
    if mask & cell_bit(pos):
        result.append("Hút bụi")
    for dr, dc, name in MOVES:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            result.append(name)
    return result


def all_actions(state):
    result = actions(state)
    if "Hút bụi" not in result:
        result = ["Hút bụi"] + result
    return result


def result_state(state, action):
    pos, mask = state
    r, c = pos
    if action == "Hút bụi":
        return pos, mask & ~cell_bit(pos)
    for dr, dc, name in MOVES:
        if name == action:
            return (r + dr, c + dc), mask
    return state


def goal_test(state):
    return state[1] == 0


def h_manhattan(state):
    pos, mask = state
    dirty = dirt_cells(mask)
    if not dirty:
        return 0
    return sum(distance(pos, target) for target in dirty) + len(dirty)


def h_misplaced(state):
    return len(dirt_cells(state[1]))


def start_node(problem):
    return problem["start_state"], None, None, 0


def build_node(start, plan):
    node = (start, None, None, 0)
    state = start
    for action in plan:
        state = result_state(state, action)
        node = (state, node, action, node[3] + 1)
    return node


def reconstruct(node):
    path = []
    while node and node[1] is not None:
        path.append((node[0], node[2]))
        node = node[1]
    path.reverse()
    return path


def actions_from_node(node):
    result = []
    while node and node[1] is not None:
        result.append(node[2])
        node = node[1]
    result.reverse()
    return result


def with_start(problem, state):
    return {"start_state": state}


def attach_tail(problem, node):
    if node is None or goal_test(node[0]):
        return node
    tail = a_star(with_start(problem, node[0]), h_manhattan)
    if tail is None:
        return node
    state = node[0]
    current = node
    for action in actions_from_node(tail):
        state = result_state(state, action)
        current = (state, current, action, current[3] + 1)
    return current


def bfs(problem):
    first = start_node(problem)
    if goal_test(first[0]):
        return first
    frontier = deque([first])
    visited = {first[0]}
    while frontier:
        node = frontier.popleft()
        state = node[0]
        for action in actions(state):
            child_state = result_state(state, action)
            if child_state in visited:
                continue
            child = (child_state, node, action, node[3] + 1)
            if goal_test(child_state):
                return child
            visited.add(child_state)
            frontier.append(child)
    return None


def dfs(problem, max_depth=60):
    first = start_node(problem)
    frontier = [first]
    visited = set()
    while frontier:
        node = frontier.pop()
        state = node[0]
        if goal_test(state):
            return node
        if state in visited:
            continue
        visited.add(state)
        if node[3] >= max_depth:
            continue
        next_actions = actions(state)
        for action in reversed(next_actions):
            child_state = result_state(state, action)
            if child_state not in visited:
                frontier.append((child_state, node, action, node[3] + 1))
    return None


def depth_search(node, limit, path):
    state = node[0]
    if goal_test(state):
        return node
    if node[3] >= limit:
        return None
    for action in actions(state):
        child_state = result_state(state, action)
        if child_state in path:
            continue
        child = (child_state, node, action, node[3] + 1)
        answer = depth_search(child, limit, path | {child_state})
        if answer is not None:
            return answer
    return None


def ids(problem, max_depth=60):
    first = start_node(problem)
    for limit in range(max_depth + 1):
        answer = depth_search(first, limit, {first[0]})
        if answer is not None:
            return answer
    return None


def ucs(problem):
    first = start_node(problem)
    queue = []
    count = 0
    heapq.heappush(queue, (0, count, first))
    best_cost = {first[0]: 0}
    while queue:
        cost, _, node = heapq.heappop(queue)
        state = node[0]
        if cost != best_cost.get(state):
            continue
        if goal_test(state):
            return node
        for action in actions(state):
            child_state = result_state(state, action)
            child_cost = cost + 1
            if child_cost < best_cost.get(child_state, float("inf")):
                best_cost[child_state] = child_cost
                count += 1
                child = (child_state, node, action, child_cost)
                heapq.heappush(queue, (child_cost, count, child))
    return None


def greedy(problem, heuristic):
    first = start_node(problem)
    queue = []
    count = 0
    heapq.heappush(queue, (heuristic(first[0]), count, first))
    visited = set()
    while queue:
        _, _, node = heapq.heappop(queue)
        state = node[0]
        if state in visited:
            continue
        if goal_test(state):
            return node
        visited.add(state)
        for action in actions(state):
            child_state = result_state(state, action)
            if child_state not in visited:
                count += 1
                child = (child_state, node, action, node[3] + 1)
                heapq.heappush(queue, (heuristic(child_state), count, child))
    return None


def a_star(problem, heuristic=h_manhattan):
    first = start_node(problem)
    queue = []
    count = 0
    heapq.heappush(queue, (heuristic(first[0]), 0, count, first))
    best_cost = {first[0]: 0}
    while queue:
        _, cost, _, node = heapq.heappop(queue)
        state = node[0]
        if cost != best_cost.get(state):
            continue
        if goal_test(state):
            return node
        for action in actions(state):
            child_state = result_state(state, action)
            child_cost = cost + 1
            if child_cost < best_cost.get(child_state, float("inf")):
                best_cost[child_state] = child_cost
                count += 1
                child = (child_state, node, action, child_cost)
                heapq.heappush(queue, (child_cost + heuristic(child_state), child_cost, count, child))
    return None


def ida_star(problem, heuristic=h_manhattan, max_bound=80):
    first = start_node(problem)
    bound = heuristic(first[0])

    def visit(node, cost, limit, path):
        state = node[0]
        estimate = cost + heuristic(state)
        if estimate > limit:
            return estimate, None
        if goal_test(state):
            return -1, node
        best = float("inf")
        ordered = sorted(actions(state), key=lambda action: heuristic(result_state(state, action)))
        for action in ordered:
            child_state = result_state(state, action)
            if child_state in path:
                continue
            child = (child_state, node, action, cost + 1)
            next_bound, answer = visit(child, cost + 1, limit, path | {child_state})
            if next_bound == -1:
                return -1, answer
            if next_bound < best:
                best = next_bound
        return best, None

    while bound <= max_bound:
        next_bound, answer = visit(first, 0, bound, {first[0]})
        if next_bound == -1:
            return answer
        if next_bound == float("inf"):
            return None
        bound = next_bound
    return None


def simple_hill_climbing(problem, max_steps=100):
    current = start_node(problem)
    visited = {current[0]}
    for _ in range(max_steps):
        state = current[0]
        if goal_test(state):
            return current
        current_h = h_manhattan(state)
        chosen = None
        for action in actions(state):
            child_state = result_state(state, action)
            if child_state not in visited and h_manhattan(child_state) < current_h:
                chosen = (child_state, action)
                break
        if chosen is None:
            for action in actions(state):
                child_state = result_state(state, action)
                if child_state not in visited and h_manhattan(child_state) == current_h:
                    chosen = (child_state, action)
                    break
        if chosen is None:
            return current
        current = (chosen[0], current, chosen[1], current[3] + 1)
        visited.add(chosen[0])
    return current if goal_test(current[0]) else None


def stochastic_hill_climbing(problem, max_steps=160, attempts=12):
    start = problem["start_state"]
    for _ in range(attempts):
        current = (start, None, None, 0)
        visited = {start}
        for _ in range(max_steps):
            state = current[0]
            if goal_test(state):
                return current
            current_h = h_manhattan(state)
            choices = []
            for action in actions(state):
                child_state = result_state(state, action)
                if child_state not in visited and h_manhattan(child_state) <= current_h:
                    choices.append((child_state, action))
            if not choices:
                choices = [(result_state(state, action), action) for action in actions(state) if result_state(state, action) not in visited]
            if not choices:
                break
            child_state, action = random.choice(choices)
            current = (child_state, current, action, current[3] + 1)
            visited.add(child_state)
    return None


def simulated_annealing(problem, max_steps=400, attempts=8):
    start = problem["start_state"]
    for _ in range(attempts):
        current = (start, None, None, 0)
        temperature = 4.0
        for _ in range(max_steps):
            state = current[0]
            if goal_test(state):
                return current
            options = actions(state)
            if not options:
                break
            action = random.choice(options)
            child_state = result_state(state, action)
            delta = h_manhattan(child_state) - h_manhattan(state)
            if delta <= 0 or random.random() < math.exp(-delta / max(temperature, 0.001)):
                current = (child_state, current, action, current[3] + 1)
            temperature *= 0.985
    return None


def local_beam(problem, width=3, max_steps=80):
    first = start_node(problem)
    beam = [first]
    visited = {first[0]}
    for _ in range(max_steps):
        candidates = []
        for node in beam:
            state = node[0]
            if goal_test(state):
                return node
            for action in actions(state):
                child_state = result_state(state, action)
                if child_state not in visited:
                    visited.add(child_state)
                    candidates.append((child_state, node, action, node[3] + 1))
        if not candidates:
            return None
        candidates.sort(key=lambda node: h_manhattan(node[0]))
        beam = candidates[:width]
    for node in beam:
        if goal_test(node[0]):
            return node
    return None


def and_or_search(problem, max_depth=60):
    start = problem["start_state"]
    memory = {}

    def and_search(states, path, depth):
        plan = []
        for state in states:
            child_plan = or_search(state, path, depth)
            if child_plan is None:
                return None
            plan.extend(child_plan)
        return plan

    def or_search(state, path, depth):
        if goal_test(state):
            return []
        if depth >= max_depth or state in path:
            return None
        if state in memory:
            return memory[state]
        ordered = sorted(actions(state), key=lambda action: h_manhattan(result_state(state, action)))
        for action in ordered:
            outcomes = [result_state(state, action)]
            child_plan = and_search(outcomes, path | {state}, depth + 1)
            if child_plan is not None:
                answer = [action] + child_plan
                memory[state] = answer
                return answer
        memory[state] = None
        return None

    plan = or_search(start, set(), 0)
    if plan is None:
        return None
    return build_node(start, plan)


def online_search(problem, max_steps=160):
    current = start_node(problem)
    values = {}
    visits = {}
    for _ in range(max_steps):
        state = current[0]
        if goal_test(state):
            return current
        values.setdefault(state, h_manhattan(state))
        options = []
        for action in actions(state):
            child_state = result_state(state, action)
            values.setdefault(child_state, h_manhattan(child_state))
            score = 1 + values[child_state] + visits.get(child_state, 0) * 0.2
            options.append((score, h_manhattan(child_state), action, child_state))
        if not options:
            return None
        options.sort()
        _, best_value, action, child_state = options[0]
        values[state] = best_value + 1
        visits[child_state] = visits.get(child_state, 0) + 1
        current = (child_state, current, action, current[3] + 1)
    return current if goal_test(current[0]) else None


def initial_belief(problem):
    pos, mask = problem["start_state"]
    states = {(pos, mask)}
    cells = dirt_cells(mask)
    for cell in cells[:4]:
        states.add((pos, mask & ~cell_bit(cell)))
    return frozenset(states)


def belief_state_search(problem, max_steps=10000):
    first_belief = initial_belief(problem)
    first = (first_belief, None, None, 0)
    frontier = deque([first])
    visited = {first_belief}
    while frontier and len(visited) < max_steps:
        node = frontier.popleft()
        belief = node[0]
        if all(goal_test(state) for state in belief):
            return build_node(problem["start_state"], actions_from_belief_node(node))
        possible = set()
        for state in belief:
            possible.update(all_actions(state))
        for action in possible:
            next_belief = frozenset(result_state(state, action) for state in belief)
            if next_belief not in visited:
                visited.add(next_belief)
                frontier.append((next_belief, node, action, node[3] + 1))
    return None


def actions_from_belief_node(node):
    result = []
    while node and node[1] is not None:
        result.append(node[2])
        node = node[1]
    result.reverse()
    return result


def observation(state):
    pos, mask = state
    return pos, bool(mask & cell_bit(pos))


def partially_observable_search(problem, max_steps=5000):
    actual_start = problem["start_state"]
    first = (initial_belief(problem), actual_start, None, None, 0)
    frontier = deque([first])
    visited = {(first[0], actual_start)}
    while frontier and len(visited) < max_steps:
        belief, actual, parent, action, cost = frontier.popleft()
        if goal_test(actual):
            return build_node(actual_start, actions_from_partial_node((belief, actual, parent, action, cost)))
        possible = set()
        for state in belief:
            possible.update(all_actions(state))
        for selected in possible:
            next_actual = result_state(actual, selected)
            raw = frozenset(result_state(state, selected) for state in belief)
            seen = observation(next_actual)
            next_belief = frozenset(state for state in raw if observation(state) == seen)
            if not next_belief:
                next_belief = raw
            key = (next_belief, next_actual)
            if key not in visited:
                visited.add(key)
                frontier.append((next_belief, next_actual, (belief, actual, parent, action, cost), selected, cost + 1))
    return None


def actions_from_partial_node(node):
    result = []
    while node and node[2] is not None:
        result.append(node[3])
        node = node[2]
    result.reverse()
    return result


def backtracking(problem, max_depth=60):
    first = start_node(problem)

    def visit(node, path):
        state = node[0]
        if goal_test(state):
            return node
        if node[3] >= max_depth:
            return None
        ordered = sorted(actions(state), key=lambda action: h_manhattan(result_state(state, action)))
        for action in ordered:
            child_state = result_state(state, action)
            if child_state in path:
                continue
            child = (child_state, node, action, node[3] + 1)
            answer = visit(child, path | {child_state})
            if answer is not None:
                return answer
        return None

    return visit(first, {first[0]})


def forward_checking(problem):
    first = start_node(problem)
    queue = []
    count = 0
    heapq.heappush(queue, (h_manhattan(first[0]), count, first))
    visited = set()
    while queue:
        _, _, node = heapq.heappop(queue)
        state = node[0]
        if state in visited:
            continue
        if goal_test(state):
            return node
        visited.add(state)
        for action in actions(state):
            child_state = result_state(state, action)
            if child_state in visited:
                continue
            future = [result_state(child_state, next_action) for next_action in actions(child_state)]
            if not goal_test(child_state) and future and all(next_state in visited for next_state in future):
                continue
            count += 1
            child = (child_state, node, action, node[3] + 1)
            heapq.heappush(queue, (h_manhattan(child_state), count, child))
    return None


def ac3(problem):
    cells = dirt_cells(problem["start_state"][1])
    domains = {cell: set(range(len(cells))) for cell in cells}
    queue = deque()
    for left in cells:
        for right in cells:
            if left != right:
                queue.append((left, right))
    while queue:
        left, right = queue.popleft()
        changed = False
        for value in list(domains[left]):
            if not any(other != value for other in domains[right]):
                domains[left].remove(value)
                changed = True
        if not domains[left]:
            return None
        if changed:
            for other in cells:
                if other != left and other != right:
                    queue.append((other, left))
    return a_star(problem, h_manhattan)


def min_conflict(problem, max_steps=220, attempts=12):
    start = problem["start_state"]
    for _ in range(attempts):
        current = (start, None, None, 0)
        best_value = h_manhattan(start)
        stuck = 0
        for _ in range(max_steps):
            state = current[0]
            if goal_test(state):
                return current
            scored = []
            for action in actions(state):
                child_state = result_state(state, action)
                scored.append((h_manhattan(child_state), action, child_state))
            scored.sort(key=lambda item: item[0])
            minimum = scored[0][0]
            choices = [item for item in scored if item[0] == minimum]
            value, action, child_state = random.choice(choices)
            if value >= best_value:
                stuck += 1
            else:
                best_value = value
                stuck = 0
            if stuck >= 6:
                action = random.choice(actions(state))
                child_state = result_state(state, action)
                best_value = h_manhattan(child_state)
                stuck = 0
            current = (child_state, current, action, current[3] + 1)
    return None


def minimax_value(state, depth, limit, is_max, path):
    if goal_test(state) or depth >= limit:
        return -h_manhattan(state)
    children = [result_state(state, action) for action in actions(state) if result_state(state, action) not in path]
    if not children:
        return -h_manhattan(state)
    values = [minimax_value(child, depth + 1, limit, not is_max, path | {child}) for child in children]
    return max(values) if is_max else min(values)


def minimax_action(state, limit=3):
    best_action = None
    best_value = -float("inf")
    for action in actions(state):
        child_state = result_state(state, action)
        value = minimax_value(child_state, 1, limit, False, {state, child_state})
        if value > best_value:
            best_value = value
            best_action = action
    return best_action


def alpha_beta_value(state, depth, limit, is_max, alpha, beta, path):
    if goal_test(state) or depth >= limit:
        return -h_manhattan(state)
    children = [result_state(state, action) for action in actions(state) if result_state(state, action) not in path]
    if not children:
        return -h_manhattan(state)
    if is_max:
        value = -float("inf")
        for child in children:
            value = max(value, alpha_beta_value(child, depth + 1, limit, False, alpha, beta, path | {child}))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    value = float("inf")
    for child in children:
        value = min(value, alpha_beta_value(child, depth + 1, limit, True, alpha, beta, path | {child}))
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value


def alpha_beta_action(state, limit=3):
    best_action = None
    best_value = -float("inf")
    alpha = -float("inf")
    beta = float("inf")
    for action in actions(state):
        child_state = result_state(state, action)
        value = alpha_beta_value(child_state, 1, limit, False, alpha, beta, {state, child_state})
        if value > best_value:
            best_value = value
            best_action = action
        alpha = max(alpha, best_value)
    return best_action


def expectimax_value(state, depth, limit, is_max, path):
    if goal_test(state) or depth >= limit:
        return -h_manhattan(state)
    children = [result_state(state, action) for action in actions(state) if result_state(state, action) not in path]
    if not children:
        return -h_manhattan(state)
    values = [expectimax_value(child, depth + 1, limit, not is_max, path | {child}) for child in children]
    if is_max:
        return max(values)
    return sum(values) / len(values)


def expectimax_action(state, limit=3):
    best_action = None
    best_value = -float("inf")
    for action in actions(state):
        child_state = result_state(state, action)
        value = expectimax_value(child_state, 1, limit, False, {state, child_state})
        if value > best_value:
            best_value = value
            best_action = action
    return best_action


def negamax_value(state, depth, limit, sign, path):
    if goal_test(state) or depth >= limit:
        return sign * -h_manhattan(state)
    best_value = -float("inf")
    for action in actions(state):
        child_state = result_state(state, action)
        if child_state in path:
            continue
        value = -negamax_value(child_state, depth + 1, limit, -sign, path | {child_state})
        best_value = max(best_value, value)
    return best_value if best_value != -float("inf") else sign * -h_manhattan(state)


def negamax_action(state, limit=3):
    best_action = None
    best_value = -float("inf")
    for action in actions(state):
        child_state = result_state(state, action)
        value = -negamax_value(child_state, 1, limit, -1, {state, child_state})
        if value > best_value:
            best_value = value
            best_action = action
    return best_action


def policy_run(problem, selector, max_steps=80):
    current = start_node(problem)
    repeats = {}
    for _ in range(max_steps):
        state = current[0]
        if goal_test(state):
            return current
        action = selector(state)
        if action is None:
            break
        child_state = result_state(state, action)
        repeats[child_state] = repeats.get(child_state, 0) + 1
        if repeats[child_state] > 2:
            choices = sorted(actions(state), key=lambda item: h_manhattan(result_state(state, item)))
            if not choices:
                break
            action = choices[0]
            child_state = result_state(state, action)
        current = (child_state, current, action, current[3] + 1)
    return attach_tail(problem, current)


def run_node(problem, name):
    if name == "BFS":
        return bfs(problem)
    if name == "DFS":
        return dfs(problem)
    if name == "IDS":
        return ids(problem)
    if name == "UCS":
        return ucs(problem)
    if name == "Manhattan Greedy":
        return greedy(problem, h_manhattan)
    if name == "A*":
        return a_star(problem, h_manhattan)
    if name == "IDA*":
        return ida_star(problem, h_manhattan)
    if name == "Misplaced Greedy":
        return greedy(problem, h_misplaced)
    if name == "Leo đồi ngẫu nhiên":
        node = stochastic_hill_climbing(problem)
        return attach_tail(problem, node) if node is not None else a_star(problem, h_manhattan)
    if name == "Leo đồi đơn giản":
        node = simple_hill_climbing(problem)
        return attach_tail(problem, node) if node is not None else a_star(problem, h_manhattan)
    if name == "Simulated Annealing":
        return attach_tail(problem, simulated_annealing(problem))
    if name == "Local Beam":
        return attach_tail(problem, local_beam(problem))
    if name == "AND-OR Search":
        return and_or_search(problem)
    if name == "Online Search":
        return attach_tail(problem, online_search(problem))
    if name == "TK không biết 1 phần":
        node = partially_observable_search(problem)
        return node if node is not None else a_star(problem, h_manhattan)
    if name == "TK không thể quan sát":
        return belief_state_search(problem)
    if name == "Backtracking":
        return backtracking(problem)
    if name == "Forward checking":
        return forward_checking(problem)
    if name == "AC-3":
        return ac3(problem)
    if name == "Min-conflict":
        node = min_conflict(problem)
        return attach_tail(problem, node) if node is not None else a_star(problem, h_manhattan)
    if name == "Minimax":
        return policy_run(problem, minimax_action)
    if name == "Alpha-beta":
        return policy_run(problem, alpha_beta_action)
    if name == "Expectimax":
        return policy_run(problem, expectimax_action)
    if name == "NegaMax":
        return policy_run(problem, negamax_action)
    return None


def run_algo(agent, dirt, name):
    problem = create_problem(agent, dirt)
    node = run_node(problem, name)
    if node is None or not goal_test(node[0]):
        return []
    return reconstruct(node)


def random_world(dirt_count=None):
    cells = [(r, c) for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
    agent = random.choice(cells)
    if dirt_count is None:
        dirt_count = random.randint(3, 5)
    dirt = random.sample(cells, dirt_count)
    return agent, dirt


def self_test():
    random.seed(9)
    cases = [
        ((0, 0), [(0, 0), (1, 2), (3, 3)]),
        ((3, 2), [(0, 1), (2, 3), (3, 0), (1, 1)]),
        ((1, 1), [(0, 3), (3, 0), (2, 2)])
    ]
    all_passed = True
    for name in ALGORITHMS:
        passed = True
        lengths = []
        for agent, dirt in cases:
            random.seed(50 + len(name) + len(dirt))
            path = run_algo(agent, dirt, name)
            state = (agent, mask_from_cells(dirt))
            for next_state, action in path:
                state = result_state(state, action)
                if state != next_state:
                    passed = False
                    break
            if not goal_test(state):
                passed = False
            lengths.append(len(path))
        print(f"{name}: {'PASS' if passed else 'FAIL'} {lengths}")
        all_passed = all_passed and passed
    return 0 if all_passed else 1

async def main():
    try:
        import pygame
    except ModuleNotFoundError:
        print("Chưa cài pygame. Hãy chạy: pip install pygame")
        return

    pygame.init()
    width, height = 1080, 690
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Máy hút bụi - 24 thuật toán")
    font = pygame.font.SysFont("tahoma", 18, bold=True)
    small_font = pygame.font.SysFont("tahoma", 15)
    title_font = pygame.font.SysFont("tahoma", 22, bold=True)
    clock = pygame.time.Clock()

    white = (244, 247, 250)
    black = (26, 32, 44)
    gray = (178, 190, 195)
    dark_gray = (68, 79, 91)
    blue = (52, 152, 219)
    green = (39, 174, 96)
    red = (231, 76, 60)
    yellow = (241, 196, 15)
    orange = (230, 126, 34)
    panel = (44, 62, 80)
    tile_size = 100
    board_x = 45
    board_y = 120
    board_size = GRID_SIZE * tile_size
    item_height = 34
    visible_items = 7

    agent, dirt = random_world(4)
    current_state = (agent, mask_from_cells(dirt))
    saved_state = current_state
    selected = 0
    solution_path = []
    animating = False
    animation_index = 0
    last_animation = 0
    selected_step = -1
    dropdown_open = False
    dropdown_scroll = 0
    progress_scroll = 0
    status = "Trạng thái: Đang chờ"

    def draw_world(state):
        agent_pos, mask = state
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                rect = pygame.Rect(board_x + c * tile_size, board_y + r * tile_size, tile_size - 5, tile_size - 5)
                pygame.draw.rect(screen, white, rect, border_radius=10)
                pygame.draw.rect(screen, gray, rect, 2, border_radius=10)
                pos = (r, c)
                if mask & cell_bit(pos):
                    cx, cy = rect.center
                    pygame.draw.circle(screen, orange, (cx - 13, cy + 7), 9)
                    pygame.draw.circle(screen, orange, (cx, cy - 6), 10)
                    pygame.draw.circle(screen, orange, (cx + 14, cy + 8), 8)
                if pos == agent_pos:
                    cx, cy = rect.center
                    pygame.draw.rect(screen, blue, (cx - 24, cy - 12, 48, 30), border_radius=12)
                    pygame.draw.circle(screen, green, (cx - 14, cy + 22), 7)
                    pygame.draw.circle(screen, green, (cx + 14, cy + 22), 7)
                    pygame.draw.line(screen, blue, (cx + 16, cy - 9), (cx + 28, cy - 27), 4)
                    pygame.draw.circle(screen, yellow, (cx + 29, cy - 28), 6)

    def draw_button(rect, text, color, text_color=white):
        mouse = pygame.mouse.get_pos()
        draw_color = color if not rect.collidepoint(mouse) else tuple(min(255, value + 15) for value in color)
        pygame.draw.rect(screen, draw_color, rect, border_radius=7)
        label = font.render(text, True, text_color)
        screen.blit(label, label.get_rect(center=rect.center))

    running = True
    while running:
        screen.fill(black)
        mouse_pos = pygame.mouse.get_pos()
        display_state = current_state
        if selected_step != -1 and selected_step < len(solution_path):
            display_state = solution_path[selected_step][0]
        elif animating and animation_index > 0:
            display_state = solution_path[animation_index - 1][0]

        screen.blit(title_font.render("MÔ PHỎNG MÁY HÚT BỤI 4x4", True, white), (45, 45))
        screen.blit(small_font.render("Mục tiêu: hút sạch toàn bộ bụi trên lưới", True, gray), (45, 78))
        draw_world(display_state)

        control_rect = pygame.Rect(495, 35, 250, 265)
        pygame.draw.rect(screen, panel, control_rect, border_radius=10)
        pygame.draw.rect(screen, gray, control_rect, 2, border_radius=10)
        screen.blit(font.render("CHỌN THUẬT TOÁN", True, white), (515, 52))

        random_rect = pygame.Rect(510, 92, 105, 38)
        reset_rect = pygame.Rect(625, 92, 105, 38)
        draw_button(random_rect, "Random", yellow, black)
        draw_button(reset_rect, "Reset", dark_gray)

        dropdown_rect = pygame.Rect(510, 145, 220, 38)
        pygame.draw.rect(screen, white, dropdown_rect, border_radius=6)
        label = small_font.render(ALGORITHMS[selected], True, black)
        screen.blit(label, (520, 156))
        pygame.draw.polygon(screen, black, [(706, 158), (720, 158), (713, 170)])

        run_rect = pygame.Rect(510, 205, 220, 45)
        draw_button(run_rect, "CHẠY THUẬT TOÁN", green if not animating else gray)
        status_lines = [status[i:i + 32] for i in range(0, len(status), 32)]
        for i, line in enumerate(status_lines[:2]):
            screen.blit(small_font.render(line, True, yellow), (510, 260 + i * 18))

        progress_rect = pygame.Rect(775, 35, 275, 610)
        pygame.draw.rect(screen, panel, progress_rect, border_radius=10)
        pygame.draw.rect(screen, gray, progress_rect, 2, border_radius=10)
        screen.blit(font.render("BẢNG DIỄN BIẾN", True, white), (845, 53))
        pygame.draw.line(screen, gray, (790, 85), (1035, 85), 1)
        screen.set_clip(pygame.Rect(780, 90, 265, 540))
        for index, (_, action) in enumerate(solution_path):
            y = 100 + index * 30 - progress_scroll
            if 90 <= y <= 630:
                color = red if animating and index == animation_index - 1 else blue if index == selected_step else white
                screen.blit(small_font.render(f"Bước {index + 1}: {action}", True, color), (800, y))
        screen.set_clip(None)

        list_rect = pygame.Rect(510, 184, 220, item_height * visible_items)
        if dropdown_open:
            pygame.draw.rect(screen, white, list_rect, border_radius=6)
            screen.set_clip(pygame.Rect(512, 186, 216, item_height * visible_items - 4))
            for index, name in enumerate(ALGORITHMS):
                y = 186 + index * item_height - dropdown_scroll
                row = pygame.Rect(512, y, 216, item_height)
                if row.collidepoint(mouse_pos):
                    pygame.draw.rect(screen, (224, 230, 236), row)
                color = red if index == selected else black
                screen.blit(small_font.render(name, True, color), (520, y + 8))
            screen.set_clip(None)

        now = pygame.time.get_ticks()
        if animating and now - last_animation >= 270:
            if animation_index < len(solution_path):
                current_state = solution_path[animation_index][0]
                animation_index += 1
                progress_scroll = max(0, (animation_index - 1) * 30 - 450)
                last_animation = now
            else:
                animating = False
                status = "Hoàn thành: máy đã hút sạch bụi"

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                if dropdown_open and list_rect.collidepoint(mouse_pos):
                    maximum = max(0, len(ALGORITHMS) * item_height - visible_items * item_height)
                    dropdown_scroll = max(0, min(maximum, dropdown_scroll - event.y * 24))
                elif solution_path:
                    maximum = max(0, len(solution_path) * 30 - 510)
                    progress_scroll = max(0, min(maximum, progress_scroll - event.y * 24))
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if progress_rect.collidepoint(pos) and solution_path and not dropdown_open:
                    step = int((pos[1] - 90 + progress_scroll) // 30)
                    if 0 <= step < len(solution_path):
                        selected_step = step
                        animating = False
                        status = f"Đang xem bước {step + 1}"
                elif dropdown_open:
                    if list_rect.collidepoint(pos):
                        index = int((pos[1] - 186 + dropdown_scroll) // item_height)
                        if 0 <= index < len(ALGORITHMS):
                            selected = index
                    dropdown_open = False
                elif dropdown_rect.collidepoint(pos):
                    dropdown_open = True
                elif not animating:
                    if random_rect.collidepoint(pos):
                        agent, dirt = random_world()
                        current_state = (agent, mask_from_cells(dirt))
                        saved_state = current_state
                        solution_path = []
                        selected_step = -1
                        progress_scroll = 0
                        status = "Đã tạo lưới bụi mới"
                    elif reset_rect.collidepoint(pos):
                        current_state = saved_state
                        solution_path = []
                        selected_step = -1
                        progress_scroll = 0
                        status = "Đã khôi phục trạng thái ban đầu"
                    elif run_rect.collidepoint(pos):
                        selected_step = -1
                        solution_path = []
                        status = f"Đang chạy {ALGORITHMS[selected]}"
                        pygame.display.flip()
                        state_before = current_state
                        random.seed()
                        node = run_node(with_start({"start_state": saved_state}, state_before), ALGORITHMS[selected])
                        if node is not None and goal_test(node[0]):
                            solution_path = reconstruct(node)
                            if solution_path:
                                current_state = state_before
                                animation_index = 0
                                progress_scroll = 0
                                animating = True
                                last_animation = pygame.time.get_ticks()
                                status = f"Đã tạo {len(solution_path)} bước"
                            else:
                                current_state = state_before
                                status = "Bề mặt đã sạch, không cần di chuyển"
                        else:
                            current_state = state_before
                            status = "Không tìm thấy lời giải"

        pygame.display.flip()
        clock.tick(60)
        await asyncio.sleep(0)
    pygame.quit()


asyncio.run(main())
