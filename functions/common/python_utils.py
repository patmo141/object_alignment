# Copyright (C) 2019 Christopher Gearhart
# chris@bblanimation.com
# http://bblanimation.com/
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# System imports
import marshal
import itertools
import operator
import hashlib
import re
import sys
import zlib
import binascii
from io import StringIO

# Blender imports
# NONE!

# Module imports
# NONE!


#################### LISTS ####################


# USE EXAMPLE: idfun=(lambda x: x.lower()) so that it ignores case
# https://www.peterbe.com/plog/uniqifiers-benchmark
def uniquify(seq:iter, idfun=None):
    # order preserving
    if idfun is None:
        def idfun(x):
            return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)
    return result


# Not order preserving
def uniquify1(seq:iter):
    keys = {}
    for e in seq:
        keys[e] = 1
    return list(keys.keys())

def uniquify2(seq:list, innerType:type=list):
    return [innerType(x) for x in set(tuple(x) for x in seq)]


# efficient removal from list if unordered
def remove_item(ls:list, item):
    try:
        i = ls.index(item)
    except ValueError:
        return False
    ls[-1], ls[i] = ls[i], ls[-1]
    ls.pop()
    return True


# code from https://stackoverflow.com/questions/1518522/python-most-common-element-in-a-list
def most_common(L:list):
    """ find the most common item in a list """
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print "SL:", SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))

    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print "item %r, count %r, minind %r" % (item, count, min_index)
        return count, -min_index

    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]


def check_equal(lst:list):
    """ verifies that all items in list are the same """
    return lst.count(lst[0]) == len(lst)


def is_unique(lst:list):
    """ verifies that all items in list are unique """
    try:
        import numpy as np
        return np.unique(lst).size == len(lst)
    # in case the user is running Ubuntu without numpy installed
    except ImportError:
        return len(lst) == len(set(lst))


#################### STRINGS ####################


def cap(string:str, max_len:int):
    """ return string whose length does not exceed max_len """
    return string[:max_len] if len(string) > max_len else string


def rreplace(s:str, old:str, new:str, occurrence:int=1):
    """ replace limited occurences of 'old' with 'new' in string starting from end """
    li = s.rsplit(old, occurrence)
    return new.join(li)


def hash_str(string:str):
    return hashlib.md5(string.encode()).hexdigest()


def compress_str(string:str):
    compressed_str = zlib.compress(string.encode("utf-8"))
    compressed_str = binascii.hexlify(compressed_str)
    return compressed_str.decode()


def decompress_str(string:str):
    decompressed_str = binascii.unhexlify(string)
    decompressed_str = zlib.decompress(decompressed_str)
    return decompressed_str.decode()


def str_to_bool(s:str):
    if s.lower() == "true":
        return True
    elif s.lower() == "false":
        return False
    else:
        raise ValueError("String '%(s)s' could not be evaluated as a bool" % locals())


#################### OTHER ####################


def deepcopy(object):
    """ efficient way to deepcopy marshal loadable object """
    marshal_obj = marshal.dumps(object)
    new_obj = marshal.loads(marshal_obj)
    return new_obj


def confirm_list(object):
    """ if single item passed, convert to list """
    if type(object) not in (list, tuple):
        object = [object]
    return object


def confirm_iter(object):
    """ if single item passed, convert to list """
    try:
        iter(object)
    except TypeError:
        object = [object]
    return object


def camel_to_snake_case(str):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', str)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class Suppressor(object):
    """ silence function and prevent exceptions """
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self
    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout
        if type is not None:
            # Uncomment next line to do normal exception handling
            # raise
            pass
    def write(self, x):
        pass
# with Suppressor():
#     do_something()


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout
# with Capturing() as output:
#     do_something()
