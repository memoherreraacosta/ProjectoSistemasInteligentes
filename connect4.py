import random
import pygame
import math
import threading
import problema2
import numpy as np



def create_board():
	return np.zeros((ROW_COUNT, COLUMN_COUNT)) # A board


def drop_piece(board, row, col, piece):
	board[row][col] = piece


def is_valid_location(board, col):
	return board[ROW_COUNT-1][col] == 0


def get_next_open_row(board, col):
	return next(
		r
		for r in range(ROW_COUNT)
		if board[r][col] == 0
	)


def winning_move(board, piece):
	# Check horizontal locations for win
	for c in range(COLUMN_COUNT-3):
		for r in range(ROW_COUNT):
			if (board[r][c] == piece and
			  board[r][c+1] == piece and
			  board[r][c+2] == piece and
			  board[r][c+3] == piece):
				return True
	# Check vertical locations for win
	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT-3):
			if (board[r][c] == piece and
			  board[r+1][c] == piece and
			  board[r+2][c] == piece and
			  board[r+3][c] == piece):
				return True
	# Check positively sloped diaganols
	for c in range(COLUMN_COUNT-3):
		for r in range(ROW_COUNT-3):
			if (board[r][c] == piece and
			  board[r+1][c+1] == piece and
			  board[r+2][c+2] == piece and
			  board[r+3][c+3] == piece):
				return True
	# Check negatively sloped diaganols
	for c in range(COLUMN_COUNT-3):
		for r in range(3, ROW_COUNT):
			if (board[r][c] == piece and
			  board[r-1][c+1] == piece and
			  board[r-2][c+2] == piece and
			  board[r-3][c+3] == piece):
				return True


def evaluate_window(window, piece):
	score = 0
	opp_piece = PLAYER_PIECE
	if piece == PLAYER_PIECE:
		opp_piece = AI_PIECE
	if window.count(piece) == 4:
		score += 100
	elif window.count(piece) == 3 and window.count(EMPTY) == 1:
		score += 5
	elif window.count(piece) == 2 and window.count(EMPTY) == 2:
		score += 2
	if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
		score -= 4
	return score


def score_position(board, piece):
	score = 0
	## Score center column
	center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
	center_count = center_array.count(piece)
	score += center_count * 3
	## Score Horizontal
	for r in range(ROW_COUNT):
		row_array = [int(i) for i in list(board[r, :])]
		for c in range(COLUMN_COUNT-3):
			window = row_array[c:c+WINDOW_LENGTH]
			score += evaluate_window(window, piece)
	## Score Vertical
	for c in range(COLUMN_COUNT):
		col_array = [int(i) for i in list(board[:, c])]
		for r in range(ROW_COUNT-3):
			window = col_array[r:r+WINDOW_LENGTH]
			score += evaluate_window(window, piece)
	## Score posiive sloped diagonal
	for r in range(ROW_COUNT-3):
		for c in range(COLUMN_COUNT-3):
			window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
			score += evaluate_window(window, piece)
	for r in range(ROW_COUNT-3):
		for c in range(COLUMN_COUNT-3):
			window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
			score += evaluate_window(window, piece)
	return score


def is_terminal_node(board):
	return (
		winning_move(board, PLAYER_PIECE) or
		winning_move(board, AI_PIECE) or
		len(get_valid_locations(board)) == 0
	)


def minimax(board, depth, alpha, beta, maximizingPlayer):
	valid_locations = get_valid_locations(board)
	is_terminal = is_terminal_node(board)
	if depth == 0 or is_terminal:
		if is_terminal:
			if winning_move(board, AI_PIECE):
				return (None, 100000000000000)
			elif winning_move(board, PLAYER_PIECE):
				return (None, -10000000000000)
			else: # Game is over, no more valid moves
				return (None, 0)
		else: # Depth is zero
			return (None, score_position(board, AI_PIECE))
	if maximizingPlayer:
		value = -math.inf
		column = random.choice(valid_locations)
		for col in valid_locations:
			row = get_next_open_row(board, col)
			b_copy = board.copy()
			drop_piece(b_copy, row, col, AI_PIECE)
			new_score = minimax(b_copy, depth-1, alpha, beta, False)[1]
			if new_score > value:
				value = new_score
				column = col
			alpha = max(alpha, value)
			if alpha >= beta:
				break
		return column, value
	else: 
		# Minimizing player
		value = math.inf
		column = random.choice(valid_locations)
		for col in valid_locations:
			row = get_next_open_row(board, col)
			b_copy = board.copy()
			drop_piece(b_copy, row, col, PLAYER_PIECE)
			new_score = minimax(b_copy, depth-1, alpha, beta, True)[1]
			if new_score < value:
				value = new_score
				column = col
			beta = min(beta, value)
			if alpha >= beta:
				break
		return column, value

def get_valid_locations(board):
	return [
		col
		for col in range(COLUMN_COUNT)
		if is_valid_location(board, col)
	]


def pick_best_move(board, piece):
	valid_locations = get_valid_locations(board)
	best_score = -10000
	best_col = random.choice(valid_locations)
	for col in valid_locations:
		row = get_next_open_row(board, col)
		temp_board = board.copy()
		drop_piece(temp_board, row, col, piece)
		score = score_position(temp_board, piece)
		if score > best_score:
			best_score = score
			best_col = col
	return best_col


def draw_board(board):
	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT):
			pygame.draw.rect(
				screen,
				YELLOW,
				(c*SQUARESIZE, r*SQUARESIZE+SQUARESIZE, SQUARESIZE, SQUARESIZE)
			)
			pygame.draw.circle(screen, WHITE,
				(int(c*SQUARESIZE+SQUARESIZE/2), int(r*SQUARESIZE+SQUARESIZE+SQUARESIZE/2)),
				RADIUS
			)
	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT):		
			if board[r][c] == PLAYER_PIECE:
				pygame.draw.circle(
					screen,
					RED,
					(int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)),
					RADIUS
				)
			elif board[r][c] == AI_PIECE: 
				pygame.draw.circle(
					screen,
					BLACK,
					(int(c*SQUARESIZE+SQUARESIZE/2), height-int(r*SQUARESIZE+SQUARESIZE/2)),
					RADIUS
				)
	pygame.display.update()


YELLOW = (245,245,0)
WHITE = (250,250,250)
GRAY = (211,211,211)
RED = (200,0,0)
BLACK = (0,0,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

# Player number
PLAYER = 0
AI_MINIMAX = 1

EMPTY = 0
PLAYER_PIECE = 1

AI_PIECE = 2
WINDOW_LENGTH = 4
# Initial turn player
turn = PLAYER
# Initial coin position
posx = 50
# Size for the window
SQUARESIZE = 100
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE
# Change to: (1200, 700) to split the game view
#            (width, height) to show the board only
size = (width, height)
RADIUS = int(SQUARESIZE/2 - 5)


#------- Init the game -------#
pygame.init()
pygame.mixer.init()
game_over = False
pygame.display.set_caption('Conecta 4')
#sound = pygame.mixer.Sound("./Chillout-downtempo-music-loop.mp3")
#sound.play(-1)
#sound.set_volume(0.30)

screen = pygame.display.set_mode(size)
myfont = pygame.font.SysFont("timesnewromanbold", 75)
pygame.draw.rect(screen, WHITE, (0,0, width, SQUARESIZE))
pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
pygame.display.update()
board = create_board()
draw_board(board)

#thread_problema2 = threading.Thread(target=problema2.main)
#thread_problema2.start()

while not game_over:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			pygame.quit()
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_LEFT:
				pygame.draw.rect(screen, WHITE, (0,0, width, SQUARESIZE))
				posx = (posx - SQUARESIZE)%width
				if turn == PLAYER:
					pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
			if event.key == pygame.K_RIGHT:
				pygame.draw.rect(screen, WHITE, (0,0, width, SQUARESIZE))
				posx = (posx + SQUARESIZE)%width
				if turn == PLAYER:
					pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
			if event.key == pygame.K_DOWN:
				pygame.draw.rect(screen, WHITE, (0,0, width, SQUARESIZE))
				pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
				# Ask for Player 1 Input
				if turn == PLAYER:
					col = int(math.floor(posx/SQUARESIZE))
					if is_valid_location(board, col):
						row = get_next_open_row(board, col)
						drop_piece(board, row, col, PLAYER_PIECE)
						if winning_move(board, PLAYER_PIECE):
							label = myfont.render("Human Wins!!", True, RED)
							screen.blit(label, (20,20))
							game_over = True
						turn += 1
						turn = turn % 2
						draw_board(board)
		pygame.display.update()
	# Ask for Player 2 Input
	if turn == AI_MINIMAX and not game_over:			
		col, minimax_score = minimax(board, 5, -math.inf, math.inf, True)
		if is_valid_location(board, col):
			row = get_next_open_row(board, col)
			drop_piece(board, row, col, AI_PIECE)
			if winning_move(board, AI_PIECE):
				draw_board(board)
				pygame.draw.rect(screen, WHITE, (0,0, width, SQUARESIZE))
				label = myfont.render("AI MiniMax wins!!", True, BLACK)
				screen.blit(label, (20,20))
				game_over = True

			draw_board(board)
			turn += 1
			turn = turn % 2

	if game_over:
		#thread_problema2.join()
		#sound.stop()
		pygame.quit()
