def brent(N):
	y,c,m = random.randint(1, N-1),random.randint(1, N-1),random.randint(1, N-1)
	g,r,q = 1,1,1
	while g==1:
		x = y
		for i in range(r):
			y = ((y*y)%N+c)%N
		k = 0
		while (k<r and g==1):
			ys = y
			for i in range(min(m,r-k)):
				y = ((y*y)%N+c)%N
				q = q*(abs(x-y))%N
			g = 	(q,N)
			k = k + m
		r = r*2
	if g==N:
		while True:
			ys = ((ys*ys)%N+c)%N
			g = gcd(abs(x-ys),N)
			if g>1:
				break
	return g

def factors(n):
	if n&1 == 0:
		j=1
		n>>=1
		while n&1 == 0:
			j+=1
			n>>=1
		yield [2,j]
	while n!=1:
		g=brent(n)
		n//=g
		j=1
		while n%g==0:
			j+=1
			n//=g
		yield [g,j]

def ordre(g,n,phi):
	l=list(factors(phi))
	change=1
	while(change==1):
		change=0
		print(l)
		for d in l:
			if pow(g, phi//d[0], n) == 1:
				change=1
				phi//=d[0]
				d[1]-=1
				if d[1]==0:
					l.remove(d)
	return phi
	


import random
from collections import namedtuple


def get_primes(start, stop):
	"""Return a list of prime numbers in ``range(start, stop)``."""
	if start >= stop:
		return []
	primes = [2]
	for n in range(3, stop + 1, 2):
		for p in primes:
			if n % p == 0:
				break
		else:
			primes.append(n)
	while primes and primes[0] < start:
		del primes[0]
	return primes

def get_primes(start, stop):
	"""Return a list of prime numbers in ``range(start, stop)``."""
	if start >= stop:
		return []
	primes = [2]
	for n in range(3, stop + 1, 2):
		for p in primes:
			if n % p == 0:
				break
		else:
			primes.append(n)
	while primes and primes[0] < start:
		del primes[0]
	return primes

def make_key_pair(length):
	"""Create a public-private key pair.
	
	The key pair is generated from two random prime numbers. The argument
	``length`` specifies the bit length of the number ``n`` shared between
	the two keys: the higher, the better.
	"""
	if length < 4:
		raise ValueError('cannot generate a key of length less '
						 'than 4 (got {!r})'.format(length))
	
	# First step: find a number ``n`` which is the product of two prime
	# numbers (``p`` and ``q``). ``n`` must have the number of bits specified
	# by ``length``, therefore it must be in ``range(n_min, n_max + 1)``.
	n_min = 1 << (length - 1)
	n_max = (1 << length) - 1
	
	# The key is stronger if ``p`` and ``q`` have similar bit length. We
	# choose two prime numbers in ``range(start, stop)`` so that the
	# difference of bit lengths is at most 2.
	start = 1 << (length // 2 - 1)
	stop = 1 << (length // 2 + 1)
	primes = get_primes(start, stop)
	
	# Now that we have a list of prime number candidates, randomly select
	# two so that their product is in ``range(n_min, n_max + 1)``.
	while primes:
		p = random.choice(primes)
		primes.remove(p)
		q_candidates = [q for q in primes
						if n_min <= p * q <= n_max]
		if q_candidates:
			q = random.choice(q_candidates)
			break
	else:
		raise AssertionError("cannot find 'p' and 'q' for a key of "
							 "length={!r}".format(length))
	
	# Second step: choose a number ``e`` lower than ``(p - 1) * (q - 1)``
	# which shares no factors with ``(p - 1) * (q - 1)``.
	stop = (p - 1) * (q - 1)
	for e in range(3, stop, 2):
		if are_relatively_prime(e, stop):
			break
	else:
		raise AssertionError("cannot find 'e' with p={!r} "
							 "and q={!r}".format(p, q))
	
	# Third step: find ``d`` such that ``(d * e - 1)`` is divisible by
	# ``(p - 1) * (q - 1)``.
	for d in range(3, stop, 2):
		if d * e % stop == 1:
			break
	else:
		raise AssertionError("cannot find 'd' with p={!r}, q={!r} "
							 "and e={!r}".format(p, q, e))
	
	# That's all. We can build and return the public and private keys.
	return PublicKey(p * q, e), PrivateKey(p * q, d)


def generate2GenLargePrime(keysize=1024):
	while True:
		num = random.randrange(2**(keysize-1), 2**(keysize))
		if isPrime(num) and isPrime((num-1)//2):
			return num


	
	
import random
import sys

def is_probable_prime(n, k = 7):
	if n < 6:  # assuming n >= 0 in all cases... shortcut small cases here
		return [False, False, True, True, False, True][n]
	elif n & 1 == 0:  # should be faster than n % 2
		return False
	else:
		s, d = 0, n - 1
		while d & 1 == 0:
			s, d = s + 1, d >> 1
		# Use random.randint(2, n-2) for very large numbers
		for a in random.sample(range(2, min(n - 2, sys.maxsize)), min(n - 4, k)):
			x = pow(a, d, n)
			if x != 1 and x + 1 != n:
				for r in range(1, s):
					x = pow(x, 2, n)
					if x == 1:
						return False  # composite for sure
					elif x == n - 1:
						a = 0  # so we know loop didn't continue to end
						break  # could be strong liar, try another a
				if a:
					return False  # composite if we reached end of this loop
		return True  # probably prime if reached end of outer loop

isPrime=is_probable_prime

def generateLargePrime(keysize=1024):
	# Return a random prime number of keysize bits in size.
	while True:
		num = random.randrange(2**(keysize-1), 2**(keysize))
		if isPrime(num):
			return num

def generateDependentPrime(psize=1024, qsize=160, q=None):
	if q is None: q=generateLargePrime(qsize)
	min = 2**(psize-2*qsize-1)
	while True:
		n = random.randrange(min, 2*min)
		r = random.randrange(1, q)
		n = n*q + r
		p = n*q + 1
		if isPrime(p): break
	return (p,q)

def egcd(a, b):
	x,y,ux,vx,uy,vy = a,b,1,0,0,1
	while y != 0:
		#asset x == ux*a+vx*b
		#asset y == uy*a+vy*b
		q = x//y
		y,x = x%y,y
		ux,uy=uy,ux-q*uy
		vx,vy=vy,vx-q*vy
	return (x,ux,vx)

def gcd(a, b):
	x,y = a,b
	while y != 0:
		q = x//y
		y,x = x%y,y
	return x

def modinv(a, m):
	g, x, y = egcd(a, m)
	if g != 1:
		raise Exception('modular inverse does not exist')
	else:
		return x % m

# legendre symbol (a|m)
# note: returns m-1 if a is a non-residue, instead of -1
def legendre(a, m):
	return pow(a, (m-1) >> 1, m)

# strong probable prime
def is_sprp(n, b=2):
	d = n-1
	s = 0
	while d&1 == 0:
		s += 1
		d >>= 1
	x = pow(b, d, n)
	if x == 1 or x == n-1:
		return True
	for r in range(1, s):
		x = (x * x)%n
		if x == 1:
			return False
		elif x == n-1:
			return True
	return False

# lucas probable prime
# assumes D = 1 (mod 4), (D|n) = -1
def is_lucas_prp(n, D):
	P = 1
	Q = (1-D) >> 2
	# n+1 = 2**r*s where s is odd
	s = n+1
	r = 0
	while s&1 == 0:
		r += 1
		s >>= 1
	# calculate the bit reversal of (odd) s
	# e.g. 19 (10011) <=> 25 (11001)
	t = 0
	while s > 0:
		if s&1:
			t += 1
			s -= 1
		else:
			t <<= 1
			s >>= 1
	# use the same bit reversal process to calculate the sth Lucas number
	# keep track of q = Q**n as we go
	U = 0
	V = 2
	q = 1
	# mod_inv(2, n)
	inv_2 = (n+1) >> 1
	while t > 0:
		if t&1 == 1:
			# U, V of n+1
			U, V = ((U + V) * inv_2)%n, ((D*U + V) * inv_2)%n
			q = (q * Q)%n
			t -= 1
		else:
			# U, V of n*2
			U, V = (U * V)%n, (V * V - 2 * q)%n
			q = (q * q)%n
			t >>= 1
	# double s until we have the 2**r*sth Lucas number
	while r > 0:
			U, V = (U * V)%n, (V * V - 2 * q)%n
			q = (q * q)%n
			r -= 1
	# primality check
	# if n is prime, n divides the n+1st Lucas number, given the assumptions
	return U == 0

# primes less than 212
small_primes = set([2,  3,  5,  7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,101,103,107,109,113, 127,131,137,139,149,151,157,163,167,173, 179,181,191,193,197,199,211])

# pre-calced sieve of eratosthenes for n = 2, 3, 5, 7
indices = [1, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,101,103,107,109,113,121,127,131,137,139,143,149,151,157,163,167,169,173,179,181,187,191,193,197,199,209]

# distances between sieve values
offsets = [10, 2, 4, 2, 4, 6, 2, 6, 4, 2, 4, 6, 6, 2, 6, 4, 2, 6, 4, 6, 8, 4, 2, 4, 2, 4, 8, 6, 4, 6, 2, 4, 6, 2, 6, 6, 4, 2, 4, 6, 2, 6, 4, 2, 4, 2,10, 2]

max_int = 2147483647

# an 'almost certain' primality check
def is_prime(n):
	if n < 212:
		return n in small_primes
	for p in small_primes:
		if n%p == 0:
			return False
	# if n is a 32-bit integer, perform full trial division
	if n <= max_int:
		i = 211
		while i*i < n:
			for o in offsets:
				i += o
				if n%i == 0:
					return False
		return True
	# Baillie-PSW
	# this is technically a probabalistic test, but there are no known pseudoprimes
	if not is_sprp(n): return False
	a = 5
	s = 2
	while legendre(a, n) != n-1:
		s = -s
		a = s-a
	return is_lucas_prp(n, a)

# next prime strictly larger than n
def next_prime(n):
	if n < 2:
		return 2
	# first odd larger than n
	n = (n + 1) | 1
	if n < 212:
		while True:
			if n in small_primes:
				return n
			n += 2
	# find our position in the sieve rotation via binary search
	x = int(n%210)
	s = 0
	e = 47
	m = 24
	while m != e:
		if indices[m] < x:
			s = m
			m = (s + e + 1) >> 1
		else:
			e = m
			m = (s + e) >> 1
	i = int(n + (indices[m] - x))
	# adjust offsets
	offs = offsets[m:]+offsets[:m]
	while True:
		for o in offs:
			if is_prime(i):
				return i
			i += o

class Actor:
	def __init__(self, gm):
		(p,g,t,h)=gm
		self.gm = gm
		self.gt = pow(g,t,p)
		self.ht = pow(h,t,p)
		self.x = random.randrange(2,p-1)
		self.y = pow(g,self.x,p)

class Condidate(Actor):
	def __init__(self,gm):
		Actor.__init__(self,gm)
	def getHash(self,hash):
		(p,q,g,t,h)=self.gm
		self.gr = pow(g,random.randrange(2,p-1),p)
		return hash(self.gr)
	def getGr(self):
		return self.gr
	def setH(self,h):
		(p,q,g,t,old)=self.gm
		self.ht = pow(h,t,p)
		self.gm = (p,q,g,t,h)
	def setCs(self,cs):
		(p,g,t,h)=self.gm
		self.cs=cs
		self.py=1
		for c in cs:
			self.py=(self.py*c.y)%p

class Voter(Actor):
	def __init__(self,gm,cs):
		Actor.__init__(self,gm)
		self.cs=cs
		self.py=1
		for c in cs:
			self.py=(self.py*c.y)%p
	def setVs(self,vs):
		self.vs=vs
		self.ly=self.py
		for v in vs:
			self.ly=(self.ly*v.y)%p
	def dovote(self,i):
		(p,q,g,t,h)=self.gm
		gt=self.gt
		ht=self.ht
		ly=self.ly
		y=self.y
		cn = len(self.cs)
		if i < t-cn:
			i+=cn*random.randrange(2)
		r = random.randrange(2,p-1)
		gr = pow(g,r,p)
		yrm = (pow(ly,r,p)*pow(g,i,p))%p
		r = random.randrange(2,p-1)
		yrm *= pow(gt,r,p)
		htr = pow(ht,r,p)
		k1 = random.randrange(2,p-1)
		k2 = random.randrange(2,p-1)
		k3 = random.randrange(2,p-1)
		prouver = (pow(gr,k1,p)*pow(yrm,k2,p)*pow(htr,k3,p))%p
		joker1 = (pow(g,k1,p)*pow(ly,k2,p))%p
		joker2 = (pow(gt,k2,p)*pow(ht,k3,p))%p
		r = random.randrange(2,p-1)
		enc = (pow(g,r,p), (pow(y,r,p)*k1)%p, (pow(y,r,p)*k2)%p, (pow(y,r,p)*k3)%p)
		self.r = r
		self.k1 = k1
		self.k2 = k2
		self.k3 = k3
		return (gr, yrm, htr, joker1, joker2, prouver, enc)
	def dorevote(self, votes):
		(p,q,g,t,h)=self.gm
		gt=self.gt
		ht=self.ht
		ly=self.ly
		revotes = []
		for vote in votes:
			(gr, yrm, htr, joker1, joker2, prouver, enc) = vote
			r = random.randrange(2,p-1)
			gr = (gr*pow(g,r,p))%p
			yrm = (yrm*pow(ly,r,p))%p
			prouver = (prouver*pow(joker1,r,p))%p
			r = random.randrange(2,p-1)
			yrm = (yrm*pow(gt,r,p))%p
			htr = (htr*pow(ht,r,p))%p
			prouver = (prouver*pow(joker2,r,p))%p
			revotes += [(gr, yrm, htr, joker1, joker2, prouver, enc)]
		return revotes
	def checkRevotes(gm, votes, revotes, ks):
		(p,q,g,t,h)=self.gm
		gt=self.gt
		ht=self.ht
		ly=self.ly
		vs=self.vs
		for v, vote, revote, rkkk in zip(vs, votes, revotes, ks):
			(r, k1, k2, k3) = rkkk
			(gr, yrm, htr, joker1, joker2, prouver, enc) = vote
			if joker1 != (pow(g,k1,p)*pow(ly,k2,p))%p:
				raise Exception('vote')
			if joker2 != (pow(gt,k2,p)*pow(ht,k3,p))%p:
				raise Exception('vote')
			if enc != (pow(g,r,p), (pow(v.y,r,p)*k1)%p, (pow(v.y,r,p)*k2)%p, (pow(v.y,r,p)*k3)%p):
				raise Exception('vote')
			if (pow(gr,k1,p)*pow(yrm,k2,p)*pow(htr,k3,p))%p != prouver:
				raise Exception('vote')
			(gr, yrm, htr, joker12, joker22, prouver, enc2) = revote
			if joker1!=joker12 or joker2!=joker22 or enc!=enc2:
				return False
			if (pow(gr,k1,p)*pow(yrm,k2,p)*pow(htr,k3,p))%p != prouver:
				return False
		return True
	def remake(self,vote):
		(p,q,g,t,h)=self.gm
		gt=self.gt
		ht=self.ht
		ly=self.ly
		(gr, yrm, htr, _, _, _, _) = vote
		k1 = random.randrange(2,p-1)
		k2 = random.randrange(2,p-1)
		k3 = random.randrange(2,p-1)
		prouver = (pow(gr,k1,p)*pow(yrm,k2,p)*pow(htr,k3,p))%p
		joker1 = (pow(g,k1,p)*pow(ly,k2,p))%p
		joker2 = (pow(gt,k2,p)*pow(ht,k3,p))%p
		r = random.randrange(2,p-1)
		enc = (pow(g,r,p), pow(self.y,r,p)*k1, pow(self.y,r,p)*k2, pow(self.y,r,p)*k3)
		self.r = r
		self.k1 = k1
		self.k2 = k2
		self.k3 = k3
		return (joker1, joker2, prouver, enc)

def getNoComposite(q,t,size,qt=None):
	if qt==None:
		qt=q*t
	trandq = t*random.randrange(q)
	trqmq = trandq%q
	if trqmq!=0 and q<t+trqmq:
		randt=random.randrange(1,t-1)
		if q-trqmq<=randt:
			randt+=1
	else:
		randt=random.randrange(1,t)
	result = qt*random.randrange(2**(size-1), 2**size) + trandq + randt
	assert result % t != 0 and result % q != 0
	return result

# prime power predicate

from random import randint
from fractions import gcd

def findWitness(n, k=5): # miller-rabin
    s, d = 0, n-1
    while d % 2 == 0:
        s, d = s+1, d//2
    for i in range(k):
        a = randint(2, n-1)
        x = pow(a, d, n)
        if x == 1 or x == n-1: continue
        for r in range(1, s):
            x = (x * x) % n
            if x == 1: return a
            if x == n-1: break
        else: return a
    return 0

# returns p,k such that n=p**k, or 0,0
# assumes n is an integer greater than 1
def primePower(n):
    def checkP(n, p):
        k = 0
        while n > 1 and n % p == 0:
            n, k = n // p, k + 1
        if n == 1: return p, k
        else: return 0, 0
    if n % 2 == 0: return checkP(n, 2)
    q = n
    while True:
        a = findWitness(q)
        if a == 0: return checkP(n, q)
        d = gcd(pow(a,q,n)-a, q)
        if d == 1 or d == q: return 0, 0
        q = d

from math import log

def delta(N, q):
	(p,a) = primePower(q)
	if p==0: return False
	return (p, a, (N+q)//(2*q))

def genFactorized(size):
	factors = {}
	product = 1
	N = 1<<size
	while product.bit_length()<size:
		b = N.bit_length()
		while True:
			j = random.randrange(1, b)
			min = 1<<j
			q = random.randrange(min, min<<1)
			if N < q : continue
			d = delta(N, q)
			if d == False : continue
			(p, a, v) = d
			if random.uniform(0,1)*log(N)/log(p) < v * min : break
		N//=q
		product *= q
		if p in factors: factors[p]+=a
		else : factors[p]=a
	return product, factors

def genFactorizedBiaised(size):
	factors = []
	product = 1
	while product.bit_length()<size:
		x = random.randrange(256, 512)
		factors += [x]
		product *= x
	return product, factors

def getNoComposite(q,t,size,qt=None):
	if qt==None:
		qt=q*t
	trandq = t*random.randrange(q)
	trqmq = trandq%q
	if trqmq!=0 and q<t+trqmq:
		randt=random.randrange(1,t-1)
		if q-trqmq<=randt:
			randt+=1
	else:
		randt=random.randrange(1,t)
	return qt*random.randrange(2**(size-1), 2**size) + trandq + randt


def createElGamalSystem(psize=1024, qsize=160, c=100):
	q=generateLargePrime(qsize)
	t=next_prime(c)
	qt = q*t
	qsize+=t.bit_length()
	while True:
		n = getNoComposite(q,t,psize-qsize,qt)
		p = qt*n+1
		if isPrime(p):
			break
	while True:
		g = random.randrange(2,p-1)
		if pow(g, (p-1)//q, p) != 1 && pow(g, (p-1)//t, p) != 1:
			break
	g = pow(g, (p-1)//qt, p)
	return (p,q,g,t,0)

def getH(cs, hash):
	hs=[]
	h=1
	for c in cs:
		hs+=[c.getHash(hash)]
	for c,hashed in zip(cs,hs):
		gr = c.getGr()
		if hash(gr) != hashed:
			raise Exception("hash doesn't correspond")
		h*=gr
	return h

def createCondidates(gm,hash,nc=100):
	cs = []
	for i in range(nc):
		cs += [Condidate(gm)]
	h=getH(cs,hash)
	py=1
	(p,q,g,t,old)=gm
	for c in cs:
		py=(py*c.y)%p
		c.setCs(cs)
		c.setH(h)
	gm=(p,q,g,t,h)
	return (cs,gm,py)

def createVoters(gm,cs,py,nv=100):
	vs = []
	ly = py
	for i in range(nv):
		vs += [Voter(gm,cs)]
	for v in vs:
		ly=(ly*v.y)%p
		v.setVs(vs)
	return (vs,ly)

def dovotes(vs,cn=100):
	votes = []
	for v in vs:
		votes += [v.dovote(random.randrange(cn))]
	return votes

def publishKs(vs):
	ks=[]
	for v in vs:
		ks+=[(v.r, v.k1, v.k2, v.k3)]
	return ks

def checkRevotes(gm, votes, revotes, gt, ht, ly, vs):
	(p,q,g,t,h)=gm
	ks = publishKs(vs)
	for v, vote, revote, rkkk in zip(vs, votes, revotes, ks):
		(r, k1, k2, k3) = rkkk
		(gr, yrm, htr, joker1, joker2, prouver, enc) = vote
		if joker1 != (pow(g,k1,p)*pow(ly,k2,p))%p:
			raise Exception('vote')
		if joker2 != (pow(gt,k2,p)*pow(ht,k3,p))%p:
			raise Exception('vote')
		if enc != (pow(g,r,p), (pow(v.y,r,p)*k1)%p, (pow(v.y,r,p)*k2)%p, (pow(v.y,r,p)*k3)%p):
			raise Exception('vote')
		if (pow(gr,k1,p)*pow(yrm,k2,p)*pow(htr,k3,p))%p != prouver:
			raise Exception('vote')
		(gr, yrm, htr, _, _, prouver, _) = revote
		if (pow(gr,k1,p)*pow(yrm,k2,p)*pow(htr,k3,p))%p != prouver:
			return False
	return True

def newVotes(gm, votes, revotes, gt, ht, ly, vs):
	for v, vote, revote in zip(vs, votes, revotes):
		(_, _, _, joker1, joker2, _, enc) = vote
		(_, _, _, joker12, joker22, _, enc2) = revote
		if joker1!=joker12 or joker2!=joker22 or enc!=enc2:
			return votes
	if checkRevotes(gm, votes, revotes, gt, ht, ly, vs):
		votes = revotes
	remades = []
	newvotes = []
	for v,vote in zip(vs,votes):
		remade=v.remake(vote)
		remades+=[remade]
		(gr, yrm, htr, _, _, _, _) = vote
		(joker1, joker2, prouver, enc) = remade
		newvotes+=[(gr, yrm, htr, joker1, joker2, prouver, enc)]
	return newvotes

gm = createElGamalSystem()
(cs, gm, py) = createCondidates(gm, (lambda x: x))
(p,g,t,h)=gm
(vs,ly) = createVoters(gm, cs, py)
votes=dovotes(vs)
gt = pow(g,t,p)
ht = pow(h,t,p)
revotes = vs[0].dorevote(votes)
votes=newVotes(gm, votes, revotes, gt, ht, ly, vs)
