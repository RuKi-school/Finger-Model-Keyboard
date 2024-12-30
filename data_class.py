from dataclasses import dataclass

@dataclass
class KeyBoard:
	key		: str
	pos 	: tuple
	localpos: tuple
	put 	: bool
	framcut : int

@dataclass
class Vector:
	x 		: float
	y 		: float
	width 	: float
	height 	: float

@dataclass
class Vector2:
	x 		: float
	y 		: float