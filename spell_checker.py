import re
from collections import Counter
from queue import Queue
import nltk


def words(text): return re.findall(r'\w+', text.lower())


WORDS = Counter(words(open('big.txt').read()))


def edit_distance(corrected_word, original_word):
    """
    Calculates Damerau-Levenshtein edit distance between corrected and original word
    :param corrected_word: corrected word
    :param original_word: word which has misspellings
    :return: edit distance between corrected and original word
    """
    keyboard_graph = {'q': ['w', 'a'], 'w': ['q', 'a', 's', 'e'], 'e': ['w', 's', 'd', 'r'], 'r': ['e', 'd', 'f', 't'],
                      't': ['r', 'f', 'g', 'y'], 'y': ['t', 'g', 'h', 'u'], 'u': ['y', 'h', 'j', 'i'],
                      'i': ['u', 'j', 'k', 'o'],
                      'o': ['i', 'k', 'l', 'p'], 'p': ['o', 'l'], 'a': ['q', 'w', 's', 'z'],
                      's': ['a', 'w', 'e', 'd', 'z', 'x'],
                      'd': ['s', 'e', 'r', 'f', 'c', 'x'], 'f': ['d', 'r', 't', 'g', 'c', 'v'],
                      'g': ['f', 't', 'y', 'h', 'v', 'b'],
                      'h': ['g', 'y', 'u', 'b', 'n', 'j'], 'j': ['h', 'u', 'i', 'k', 'n', 'm'],
                      'k': ['j', 'i', 'l', 'm'],
                      'l': ['k', 'o', 'p'], 'z': ['a', 's', 'x'], 'x': ['z', 's', 'd', 'c'], 'c': ['x', 'd', 'f', 'v'],
                      'v': ['c', 'f', 'g', 'b'], 'b': ['v', 'g', 'h', 'n'], 'n': ['b', 'h', 'j', 'm'],
                      'm': ['n', 'j', 'k']}

    def calc_dist(letter1, letter2):
        """
        Calculates the shortest path from letter1 to letter2 in keyboard graph by breadth first search algorithm
        :param letter1: symbol from English alphabet
        :param letter2: symbol from English alphabet
        :return: shortest path from letter1 to letter2 in keyboard graph
        """
        q = Queue()
        q.put(letter1)
        used = set()
        used.add(letter1)
        dist = dict()
        dist[letter1] = 0
        while not q.empty():
            symbol = q.get()
            for symbol_to_go in keyboard_graph[symbol]:
                if symbol_to_go not in used:
                    used.add(symbol_to_go)
                    q.put(symbol_to_go)
                    dist[symbol_to_go] = dist[symbol] + 1
        if dist[letter2] <= 2:
            return dist[letter2]
        else:
            return 3

    distance = [[0 for i in range(len(corrected_word) + 1)] for j in range(len(original_word) + 1)]

    for i in range(len(original_word) + 1):
        for j in range(len(corrected_word) + 1):
            if min(i, j) == 0:
                distance[i][j] = max(i, j)
            else:
                deletion_cost = distance[i - 1][j] + 2

                if corrected_word[j - 1] == original_word[i - 1]:
                    # if we insert the same letter as before then add only 1 instead of 3, because this is typical error
                    insertion_cost = distance[i][j - 1] + 1
                else:
                    insertion_cost = distance[i][j - 1] + 3

                substitution_cost = distance[i - 1][j - 1] + calc_dist(original_word[i - 1], corrected_word[j - 1])

                if i > 1 and j > 1 and corrected_word[j - 1] == original_word[i - 2] \
                        and corrected_word[j - 2] == original_word[i - 1]:
                    # if we can do transposition
                    transposition_cost = distance[i - 2][j - 2] + 1  # give cost of 1 for transposition
                    distance[i][j] = min(deletion_cost, insertion_cost, substitution_cost, transposition_cost)
                else:
                    # if we can not do transposition
                    distance[i][j] = min(deletion_cost, insertion_cost, substitution_cost)

    return distance[len(original_word)][len(corrected_word)]


def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N


def correction(word, without_keyborad_layout=False):
    "Most probable spelling correction for word."
    results = known([word])

    if len(results) == 0:
        # if this word does not exist in text corpus then try edit it by 1 and 2 symbols
        results = results.union(known(edits1(word)))
        results = results.union(known(edits2(word)))

    if len(results) > 0:
        # if corrected word not begins with letter in original word then we can drop him,
        # usually nobody makes mistake at the beginning of the word
        words_to_remove = set()
        for w in results:
            if w[0] != word[0]:
                words_to_remove.add(w)
        for w in words_to_remove:
            results.remove(w)

    if len(results) == 0:
        # it means that mostly error in suffix and then correct root of the word and return it with suffix, i.e.
        # corrected root + suffix
        stemmer = nltk.stem.PorterStemmer()
        stemmed_word = stemmer.stem(word)
        if len(stemmed_word) < len(word):
            # if it is correctly stemmed
            corrected_word = correction(stemmed_word)
            if len(corrected_word) > 0:
                return corrected_word + word[len(stemmed_word):]

    if len(results) > 0 and not without_keyborad_layout:
        # if we have known edited words and in keyboard layout mode than we calculate Damerau-Levenshtein edit distance
        # for each of them and then choose most probable word, which has least edit ditance
        edit_distances = dict()
        N = sum(WORDS.values())
        for corrected_word in results:
            dist = edit_distance(corrected_word, word)
            edit_distances[dist] = edit_distances.get(dist, [])
            edit_distances[dist].append(corrected_word)
        sorted_dist = sorted(list(edit_distances.keys()))
        least_dist = sorted_dist[0]
        return max(edit_distances[least_dist], key=P)

    elif len(results) > 0 and without_keyborad_layout:
        return max(results, key=P)

    else:
        return word


def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)


def edits1(word):
    "All edits that are one edit away from `word`."
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


# Tests

def unit_tests():
    assert correction('speling') == 'spelling'  # insert
    assert correction('inconvient') == 'inconvenient'  # insert 2
    assert correction('peotry') == 'poetry'  # transpose
    assert correction('peotryy') == 'poetry'  # transpose + delete
    assert correction('word') == 'word'  # known
    assert correction('addresable') == 'addressable'  # original Norvig's solution can not handle this
    assert correction('coetect') == 'correct'
    assert correction('coetect', without_keyborad_layout=True) == 'contact'
    assert correction('cobhect') == 'connect'
    assert correction('cobhect', without_keyborad_layout=True) == 'correct'  # but r is not close to b and h as n
    assert correction('jucie') == 'juice'
    assert correction('jucie',
                      without_keyborad_layout=True) == 'julie'  # without keyboard layout it just replace c by l
    # but replacing gives a better result
    assert correction('localy') == 'locally'
    assert correction('localy', without_keyborad_layout=True) == 'local'  # this is not what we mean
    assert correction('geniva') == 'geneva'
    assert correction('geniva', without_keyborad_layout=True) == 'genius'  # here we can see that this can not be genius
    # and geneva is more fit here
    assert correction('quintessential') == 'quintessential'  # unknown
    assert words('This is a TEST.') == ['this', 'is', 'a', 'test']
    assert Counter(words('This is a test. 123; A TEST this is.')) == (
        Counter({'123': 1, 'a': 2, 'is': 2, 'test': 2, 'this': 2}))
    assert len(WORDS) == 32198
    assert sum(WORDS.values()) == 1115585
    assert WORDS.most_common(10) == [
        ('the', 79809),
        ('of', 40024),
        ('and', 38312),
        ('to', 28765),
        ('in', 22023),
        ('a', 21124),
        ('that', 12512),
        ('he', 12401),
        ('was', 11410),
        ('it', 10681)]
    assert WORDS['the'] == 79809
    assert P('quintessential') == 0
    assert 0.07 < P('the') < 0.08
    return 'unit_tests pass'


def spelltest(tests, verbose=False):
    "Run correction(wrong) on all (right, wrong) pairs; report results."
    import time
    start = time.clock()
    good, unknown = 0, 0
    n = len(tests)
    for right, wrong in tests:
        w = correction(wrong)
        good += (w == right)
        if w != right:
            unknown += (right not in WORDS)
            if verbose:
                print('correction({}) => {} ({}); expected {} ({})'
                      .format(wrong, w, WORDS[w], right, WORDS[right]))
    print('{:.0%} of {} correct ({:.0%} unknown)'
          .format(good / n, n, unknown / n))


def Testset(lines):
    "Parse 'right: wrong1 wrong2' lines into [('right', 'wrong1'), ('right', 'wrong2')] pairs."
    return [(right, wrong)
            for (right, wrongs) in (line.split(':') for line in lines)
            for wrong in wrongs.split()]


print(unit_tests())
spelltest(Testset(open('spell-testset.txt')))
