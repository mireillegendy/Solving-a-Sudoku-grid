def cross(A, B):
    return [a+b for a in A for b in B]

def grid_values(grid):
    chars = [c for c in grid if c in digits or c in '0.']
    assert len(chars) == 81
    return dict(zip(squares, chars))
def parse_grid(grid):
    values = dict((s, digits) for s in squares)
    for s, d in grid_values(grid).items():
        if d in digits and not assign(values, s, d):
            return False
        return values
def assign(values, s, d):
    other_values = values[s].replace(d, "")
    if all(eliminate(values, s, d2) for d2 in other_values):
        return values
    else:
        return False
def eliminate(values, s, d):
    if d not in values[s]:
        return values
    values[s] = values[s].replace(d, "")
    if len(values[s]) == 0:
        print("length of values[s] is 0")
        return False
    elif len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, s2, d2) for s2 in peers[s]):
            print("d2 cannot be eliminated from peers")
            return False
    for u in units[s]:
        dplaces = [s for s in u if d in values[s]]
        if len(dplaces) == 0:
            print("dplaces length is 0")
            return False
        elif len(dplaces) == 1:
            if not assign(values, dplaces[0], d):
                print("we cannot assign dplaces to values")
                return False
    return values


def solve(grid): return search(parse_grid(grid))

def search(values):
    if values is False: return False
    if all(len(values[s]) == 1 for s in squares):
        return values
    n, s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)
    return some(search(assign(values.copy(), s, d)) for d in values[s])
def some(seq):
    for e in seq:
        if e: return e
    return False

digits = '123456789'
rows = 'ABCDEFGHI'
cols = digits
squares = cross(rows, cols)
# print(squares)
unitlist = ([cross(rows,c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')])



units = dict((s, [u for u in unitlist if s in u]) for s in squares)
print(units)
peers = dict((s, set(sum(units[s], [])) - set([s])) for s in squares)
print(peers)
grid1 = "003020600900305001001806400008102900700000008006708200002609500800203009005010300"
grid = "4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......"
print(solve(grid1))
