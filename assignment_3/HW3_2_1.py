import math
header = ["OVERALL_DIAGNOSIS", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22"]

LABEL_ARRAY = [	[-1,-1, -1, -1], [-1, -1, -1, 1], [-1, -1, 1, -1], [-1, -1, 1, 1], 
				[-1, 1, -1, -1], [-1, 1, -1, 1], [-1, 1, 1, -1], [-1, 1, 1, 1], 
				[1, -1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1], [1, -1, 1, 1], 
				[1, 1, -1, -1], [1, 1, -1, 1], [1, 1, 1, -1], [1, 1, 1, 1]]

TEST_LABEL = header.index("OVERALL_DIAGNOSIS")
LABEL = header[0]

dict = []
bp = "================================================="
LEFT = 0
RIGHT = 1

def input():

	Input_Vector = []

	with open("heart_train.data", "r") as filestream:
		for line in filestream:
			currentline = line.strip()
			currentline = line.split(",")

			Input_Element = [None]*23
			for i in range(len(header)):
				if(i == TEST_LABEL):
					#print(currentline[i], "AND 0 ARE EQUAL ", int(currentline[i]) == 0)
					if(int(currentline[i]) == 0):
						Input_Element[i] = -1
						#print(Input_Element[i])
					else:
						Input_Element[i] = int(currentline[i])
				else:
					Input_Element[i] = int(currentline[i].rstrip('\r\n'))

			Input_Vector.append(Input_Element)


	return Input_Vector;

def test_input():
    Test_Vector = []

    with open("heart_test.data", "r") as filestream:
        for line in filestream:
            currentline = line.strip()
            currentline = line.split(",")

            Test_Element = [None]*23
            for i in range(len(header)):
                if(i == TEST_LABEL):
                    if(int(currentline[i]) == 0):
                        Test_Element[i] = -1
                    else:
                        Test_Element[i] = int(currentline[i])
                else:
                    Test_Element[i] = int(currentline[i])
                    
            Test_Vector.append(Test_Element)


	return Test_Vector;
'''
def test():
	Input_Vector = []
	Input_Vector = input()
	Input_Vector_Ints = range(0, len(Input_Vector))

	Test_Vector = []
	Test_Vector = test_input()



	for i in range(len(header)):
		dict.append(unique_vals(Input_Vector, i))
	

	# We need to make a root node
	root = Node("root")
	root.sp_vector = list(range(0, len(Input_Vector)))

	for i in header:
		if i != LABEL:
			root.attr2sp.append(i)

	print("-------------BUILDING TREE------------------")
	build_tree(Input_Vector, root, root)
	print("-------------PRINT TREE------------------")
	print_tree(Input_Vector, root)


	print("------------------CHECK ACCURACY-------------------------")
	print(check_accuracy(Input_Vector, root, Test_Vector), "%")
'''

def test2():
    Input_Vector = []
    Test_Vector = []
    
    Input_Vector = input()
    Test_Vector = test_input()
    #print(Input_Vector)
    List_Of_Trees = []
    List_Of_Alphas = []
    List_Of_Errors = []

    List_Of_W = [1.0/len(Input_Vector)]* len(Input_Vector)
    for i in range(len(header)):
        dict.append(unique_vals(Input_Vector, i))
    indexer = 0
        
        
    #print(w)
    
    for A in header:
		if A != "OVERALL_DIAGNOSIS":
			for B in header:
				if B != "OVERALL_DIAGNOSIS":
					for C in header:
						if C != "OVERALL_DIAGNOSIS":
							for LA in LABEL_ARRAY:
								List_Of_Trees.append(treeOneTemplate(A, B, C, Input_Vector, LA))
								List_Of_Trees.append(treeTwoTemplate(A, B, C, Input_Vector, LA))
								List_Of_Trees.append(treeThreeTemplate(A, B, C, Input_Vector, LA))
								List_Of_Trees.append(treeFourTemplate(A, B, C, Input_Vector, LA))
								List_Of_Trees.append(treeFiveTemplate(A, B, C, Input_Vector, LA))
		print(indexer)
		indexer += 1
    
   
	#lengthOfTrees = len(List_Of_Trees)
    #totalE = []
    # CALCULATE LIST OF ERRORS, ALPHAS, NEW WEIGHTS, DRAW THE BEST TREE
    for i in range(10):
        List_Of_Errors = []
        for tree in List_Of_Trees:
            List_Of_Errors.append(error_T(Input_Vector, List_Of_W, tree))
        List_Of_Alphas = compute_alphas(Input_Vector, List_Of_Errors)
        minError = min(List_Of_Errors)
        print("Error is: ", minError)
        print("Alpha is: ", List_Of_Alphas[List_Of_Errors.index(minError)])
        
        
        List_Of_W = update_weights(Input_Vector, List_Of_W, List_Of_Alphas[List_Of_Errors.index(minError)], minError, List_Of_Trees[List_Of_Errors.index(minError)])
        
        train_accuracy = check_accuracy_2(Input_Vector, List_Of_Trees[List_Of_Errors.index(minError)], Input_Vector)
        test_accuracy = check_accuracy_2(Input_Vector, List_Of_Trees[List_Of_Errors.index(minError)], Test_Vector)
        
        print("At iteration: ", i, " training accuracy is: ", train_accuracy)
        print("At iteration: ", i, " test accuracy is: ", test_accuracy)
        if i < 5:
            print_tree_2(Input_Vector, List_Of_Trees[List_Of_Errors.index(minError)])
        print(bp)
    

    
    
    #test_tree = treeOneTemplate("F2", "F3", "F4", Input_Vector, [-1,-1,1,1])
    #print(return_label_2(Input_Vector, test_tree, Input_Vector[1]))
    #print(Input_Vector[1][0])
    #W = [0.5]
    #totalE.append(error_T(Input_Vector, W, test_tree))
    #print(totalE)
    #alphas = []
    #alphas = compute_alphas(Input_Vector, totalE)
    #print(alphas)
    
    # GET MIN ERROR, FIND INDEX OF IT, USE THE INDEX FOR TREE
    # min(List_Of_Errors) Is the minimum error in the list
    # List_Of_Trees[List_Of_Errors.index(minError)] Is the tree with minimum error
    # List_Of_Alphas[List_Of_Errors.index(minError)] Is the alpha in response
    #minError = min(totalE)
    #print(minError)
    #print(totalE.index(minError))
     # Make a list of weights for each trees
     #initial_weight = 1.0 / lengthOfTrees
     #W = [initial_weight] * lengthOfTrees



# TREE ONE IS A -L-> B and A -R-> C
def treeOneTemplate(A, B, C, Input_Vector, label_array):
	# Take in the value A, A can be 0 or 1
	# With my code, it'll say split = vector of indexes. Left = 0, Right = 1
	root = Node("root")
	root.sp_vector = list(range(0, len(Input_Vector)))
	root.s_attr = A

	# SPLIT THE DATA ON A
	temp_split = split(get_temp_vector(Input_Vector, root), A)
	# Should be 0 or 1
	nodes_into_NV(Input_Vector, A, root)

	n_indexer = 0
	for n in root.node_vector:
		# This gives the A.0 the values with A = 0, and the A.1 the values where A=1
		n.sp_vector = temp_split[n_indexer]

		convert_sub_array(temp_split, root, root)
		
		# If A = 0, split on B, else, split on C
		if n.name == LEFT:
			n.s_attr = B
			temp_split_b = split(get_temp_vector(Input_Vector, n), B)
			nodes_into_NV(Input_Vector, B, n)
			#print(n.node_vector[0].name)
			# IF the left side is 0, give 0 label = [0] and 1 label = [1], else reverse
			if n.node_vector[0].name == LEFT:
				n.node_vector[0].label = label_array[0]
				n.node_vector[1].label = label_array[1]
			else: #Here, n.node_vector[0].name == 1
				n.node_vector[0].label = label_array[1]
				n.node_vector[1].label = label_array[0]

		# Here, A = 1, the right split
		elif n.name == RIGHT:
			n.s_attr = C
			temp_split_c = split(get_temp_vector(Input_Vector, n), C)
			nodes_into_NV(Input_Vector, C, n)
			if n.node_vector[0].name == LEFT:
				n.node_vector[0].label = label_array[2]
				n.node_vector[1].label = label_array[3]
			else: #Here, n.node_vector[0].name == 1
				n.node_vector[0].label = label_array[3]
				n.node_vector[1].label = label_array[2]
		n_indexer += 1

	return root

# TREE TWO IS A --L--> B --L--> C
def treeTwoTemplate(A, B, C, Input_Vector, label_array):
	root = Node("root")
	root.sp_vector = list(range(0, len(Input_Vector)))
	root.s_attr = A

	# Split the data on A
	temp_split = split(get_temp_vector(Input_Vector, root), A)
	# Should be 0 or 1
	nodes_into_NV(Input_Vector, A, root)
	n_indexer = 0
	for n in root.node_vector:
		# This gives the A.0 the values with A = 0 (to be split) and the A.1
		n.sp_vector = temp_split[n_indexer]
		convert_sub_array(temp_split, root, root)
		# IF A = 0, split on B
		if n.name == LEFT:
			n.s_attr = B
			temp_split_b = split(get_temp_vector(Input_Vector, n), B)
			nodes_into_NV(Input_Vector, B, n)
			convert_sub_array(temp_split_b, n, root)

			m_indexer = 0
			for m in n.node_vector:
				m.sp_vector = temp_split_b[m_indexer]
				if m.name == LEFT:
					m.s_attr = C
					temp_split_c = split(get_temp_vector(Input_Vector, m), C)
					nodes_into_NV(Input_Vector, C, m)

					if m.node_vector[0].label == 0:
						m.node_vector[0].label = label_array[0]
						m.node_vector[1].label = label_array[1]
					else: #C = 1
						m.node_vector[0].label = label_array[1]
						m.node_vector[1].label = label_array[0]

				elif m.name == RIGHT: #m.name == 1
					m.label = label_array[2]




				m_indexer += 1

		# If it's the right branch, take the third label
		elif n.name == RIGHT:
			n.label = label_array[3]
		n_indexer += 1


	return root

# TREE THREE IS A--L-->B --R--> C
def treeThreeTemplate(A, B, C, Input_Vector, label_array):
	root = Node("root")
	root.sp_vector = list(range(0, len(Input_Vector)))
	root.s_attr = A

	# Split the data on A
	temp_split = split(get_temp_vector(Input_Vector, root), A)
	# Should be 0 or 1
	nodes_into_NV(Input_Vector, A, root)
	n_indexer = 0
	for n in root.node_vector:
		# This gives the A.0 the values with A = 0 (to be split) and the A.1
		n.sp_vector = temp_split[n_indexer]
		convert_sub_array(temp_split, root, root)
		# IF A = 0, split on B
		if n.name == LEFT:
			n.s_attr = B
			temp_split_b = split(get_temp_vector(Input_Vector, n), B)
			nodes_into_NV(Input_Vector, B, n)
			convert_sub_array(temp_split_b, n, root)

			m_indexer = 0
			for m in n.node_vector:
				m.sp_vector = temp_split_b[m_indexer]
				if m.name == RIGHT:
					m.s_attr = C
					temp_split_c = split(get_temp_vector(Input_Vector, m), C)
					nodes_into_NV(Input_Vector, C, m)

					if m.node_vector[0].label == 0:
						m.node_vector[0].label = label_array[1]
						m.node_vector[1].label = label_array[2]
					else: #C = 1
						m.node_vector[0].label = label_array[2]
						m.node_vector[1].label = label_array[1]

				elif m.name == LEFT: #m.name == 1
					m.label = label_array[0]




				m_indexer += 1

		# If it's the right branch, take the third label
		elif n.name == RIGHT:
			n.label = label_array[3]
		n_indexer += 1


	return root

# TREE FOUR IS A --R-->B --L--> C
def treeFourTemplate(A, B, C, Input_Vector, label_array):
	root = Node("root")
	root.sp_vector = list(range(0, len(Input_Vector)))
	root.s_attr = A

	# Split the data on A
	temp_split = split(get_temp_vector(Input_Vector, root), A)
	# Should be 0 or 1
	nodes_into_NV(Input_Vector, A, root)
	n_indexer = 0
	for n in root.node_vector:
		# This gives the A.0 the values with A = 0 (to be split) and the A.1
		n.sp_vector = temp_split[n_indexer]
		convert_sub_array(temp_split, root, root)
		# IF A = 0, split on B
		if n.name == RIGHT:
			n.s_attr = B
			temp_split_b = split(get_temp_vector(Input_Vector, n), B)
			nodes_into_NV(Input_Vector, B, n)
			convert_sub_array(temp_split_b, n, root)

			m_indexer = 0
			for m in n.node_vector:
				m.sp_vector = temp_split_b[m_indexer]
				if m.name == LEFT:
					m.s_attr = C
					temp_split_c = split(get_temp_vector(Input_Vector, m), C)
					nodes_into_NV(Input_Vector, C, m)

					if m.node_vector[0].label == 0:
						m.node_vector[0].label = label_array[1]
						m.node_vector[1].label = label_array[2]
					else: #C = 1
						m.node_vector[0].label = label_array[2]
						m.node_vector[1].label = label_array[1]

				elif m.name == RIGHT: #m.name == 1
					m.label = label_array[3]




				m_indexer += 1

		# If it's the right branch, take the third label
		elif n.name == LEFT:
			n.label = label_array[0]
		n_indexer += 1


	return root

# TREE FIVE IS A --R-->B --R--> C
def treeFiveTemplate(A, B, C, Input_Vector, label_array):
	root = Node("root")
	root.sp_vector = list(range(0, len(Input_Vector)))
	root.s_attr = A

	# Split the data on A
	temp_split = split(get_temp_vector(Input_Vector, root), A)
	# Should be 0 or 1
	nodes_into_NV(Input_Vector, A, root)
	n_indexer = 0
	for n in root.node_vector:
		# This gives the A.0 the values with A = 0 (to be split) and the A.1
		n.sp_vector = temp_split[n_indexer]
		convert_sub_array(temp_split, root, root)
		# IF A = 0, split on B
		if n.name == RIGHT:
			n.s_attr = B
			temp_split_b = split(get_temp_vector(Input_Vector, n), B)
			nodes_into_NV(Input_Vector, B, n)
			convert_sub_array(temp_split_b, n, root)

			m_indexer = 0
			for m in n.node_vector:
				m.sp_vector = temp_split_b[m_indexer]
				if m.name == RIGHT:
					m.s_attr = C
					temp_split_c = split(get_temp_vector(Input_Vector, m), C)
					nodes_into_NV(Input_Vector, C, m)

					if m.node_vector[0].label == 0:
						m.node_vector[0].label = label_array[2]
						m.node_vector[1].label = label_array[3]
					else: #C = 1
						m.node_vector[0].label = label_array[3]
						m.node_vector[1].label = label_array[2]

				elif m.name == LEFT: #m.name == 1
					m.label = label_array[1]




				m_indexer += 1

		# If it's the right branch, take the third label
		elif n.name == LEFT:
			n.label = label_array[0]
		n_indexer += 1


	return root

def error_T(Input_Vector, W, root):
    errorT = 0.0
    for m in range(len(W)):
        # IF Ht(X^(M)) != y^(m), add to weight
        if return_label_2(Input_Vector, root, Input_Vector[m]) != Input_Vector[m][0]:
            errorT += W[m]
    
    return errorT
    
def compute_alphas(Input_Vector, Errors):
    alphas = []
    for i in range(len(Errors)):
        alphas.append(.5 * math.log1p((1 - Errors[i]) / Errors[i]))
        
    return alphas

def update_weights(Input_Vector, W, Alpha, Error, Tree):
    for m in range(len(W)):
        Y = Input_Vector[m][0]
        h_t = return_label_2(Input_Vector, Tree, Input_Vector[m])
        W[m] = ((W[m] * math.exp(-1.0 * Y * h_t * Alpha)) / (2 * math.sqrt(Error * (1 - Error))))
    
    return W



def unique_vals(rows, col):
		return set([row[col] for row in rows])

def type_count(rows):
	counts = {}
	for row in rows:
		label = row[TEST_LABEL]
		if label not in counts:
			counts[label] = 0
		counts[label] += 1
	return counts

# SPLITS ROWS BASED ON ATTRIBUTE
def split(rows, attribute):
	splits = []
	i = 0
	for s in dict[header.index(attribute)]:
		splits.append([])
		for j in range(len(rows)):
			if rows[j][header.index(attribute)] == s:
				splits[i].append(j)
		i += 1
	return splits
	#node


def HofY(Input_Vector, node):
	# H(Y) = - SUM prob(Y) * log2(prob(Y))
	HofY = 0
	temp_vector = get_temp_vector(Input_Vector, node)
	counts = type_count(temp_vector)
	total = 0
	#print(counts)
	for label in counts:
		total += counts[label]
	for label in counts:
		HofY -= counts[label]/total * math.log2((counts[label]/total))

	return HofY

def HofYgX(Input_Vector, node):
	HofYgX = 0
	totalX = 0
	HofYs= 0.0
	for n in node.node_vector:
		totalX += len(n.sp_vector)

	if totalX != 0:
		for n in node.node_vector:
			HofYs = HofY(Input_Vector, n)
			HofYgX += (len(n.sp_vector)/totalX)*(HofYs)

	return HofYgX


# In order to find the information gain, we need to know how many splits
# We will use
def info_gain(Input_Vector, root, node):
	return HofY(Input_Vector, root) - HofYgX(Input_Vector, node)


def best_info_gain(Input_Vector, root, node):
	best_gain = 0
	best_attribute = None
	num_columns = len(Input_Vector[0])

	print("ENTERING BEST INFO GAIN")
	# For every column
	for col in range(num_columns):
		if header[col] == LABEL:
			continue
		if header[col] in node.attr2sp:
			temp_node = Node(header[col])
			nodes_into_NV(Input_Vector, header[col], temp_node)


			temp_split = split(get_temp_vector(Input_Vector, node),header[col])
			n_indexer = 0
			for n in temp_node.node_vector:
				for i in range(len(temp_split[n_indexer])):
					#print("TSIIII: ", temp_split[n_indexer][i])
					n.sp_vector.append(node.sp_vector[temp_split[n_indexer][i]])

				n_indexer += 1

			temp_IG = info_gain(Input_Vector, root, temp_node)
			if temp_IG > best_gain:
				best_gain = temp_IG
				best_attribute = header[col]
	print("BG: ", best_gain, best_attribute)
	return best_gain, best_attribute



# Used to get a temporary array of the Input_Vector in regards to what we need (indexs)
def get_temp_vector(Input_Vector, node):
	temp_vector = []
	for i in node.sp_vector:
		temp_vector.append(Input_Vector[i])
	return temp_vector

# Split and put the new_nodes into the node vector of the previous node
def nodes_into_NV(Input_Vector, attribute, node):
	t_index = 0
	temp_vector = get_temp_vector(Input_Vector, node)
	testing = split(temp_vector, attribute)


	for s in dict[header.index(attribute)]:
		new_node = Node(s)
		new_node.sp_vector += testing[t_index]
		node.node_vector.append(new_node)
		node.s_attr = attribute
		t_index += 1


def build_tree(Input_Vector, root, node):
	IG = 0
	attribute = None
	IG, attribute = best_info_gain(Input_Vector, root, node)
	# NOW WE KNOW WHAT TO SPLIT ON
	node.s_attr = attribute
	# Split first
	#print("--------------", get_temp_vector(Input_Vector, node), "-----------------")
	temp_split = split(get_temp_vector(Input_Vector, node), attribute)
	print("-------------- TEMP SPLIT ----------------")
	nodes_into_NV(Input_Vector, attribute, node)
	print("LEN OF NV", len(node.node_vector))

	print("------------------------")
	n_indexer = 0
	for n in node.node_vector:
		# KEEP A LIST OF ATTRIBUTES WE ARE ALLOWED TO SPLIT ON
		print("Name: ", n.name)
		n.attr2sp = node.attr2sp[:]
		#print(node.attr2sp)
		n.attr2sp.remove(attribute)
		#print(n.attr2sp)

		# KEEP A LIST OF OUR VECTORS IN MAIN
		n.sp_vector = temp_split[n_indexer]

		#print(node.sp_vector)
		convert_sub_array(temp_split, node, node)
		#print(n.sp_vector)

		temp_Node = Node(n.name)
		nodes_into_NV(Input_Vector, attribute, temp_Node)
		tp_split = split(get_temp_vector(Input_Vector, n), attribute)
		convert_sub_array(tp_split, temp_Node, n)
		#print(tp_split)
		tp_index = 0

		# TO CONVERT TO ORIGINAL, I NEED TWO VECTORS, NEW SPLIT AND OLD VECTORS

		print(info_gain(Input_Vector, root, temp_Node))
		if(info_gain(Input_Vector, root, temp_Node) != HofY(Input_Vector, root)):
			build_tree(Input_Vector, root, n)

		
		n_indexer += 1

def convert_sub_array(split, temp_Node, n):
	tp_index = 0
	for tn in temp_Node.node_vector:
		# Convert to the original points
		temp_sp_vector = []
		for i in split[tp_index]:
			temp_sp_vector.append(n.sp_vector[i])

		tn.sp_vector = temp_sp_vector[:]

		tp_index += 1

def print_tree(Input_Vector, node):
	if(len(node.sp_vector) > 0):
		print("Node name: ", node.name)
		print(len(node.sp_vector))
		print("Split on: ",node.s_attr)
		for n in node.node_vector:
			print("IN: ", node.s_attr)
			print_tree(Input_Vector, n)
		
		print_label(Input_Vector, node)

def print_tree_2(Input_Vector, node):
	print("Node name: ", node.name)
	print(len(node.sp_vector))
	print("Split on: ",node.s_attr)
	if(node.s_attr != None):
		for n in node.node_vector:
			print("IN: ", node.s_attr)

			print_tree_2(Input_Vector, n)

	else:
		print("==================Label is: ", node.label, "==================")
			


def print_label(Input_Vector, node):
	temp_vector = get_temp_vector(Input_Vector, node)
	count = temp_vector[0][TEST_LABEL]
	#count = unique_vals(temp_vector, TEST_LABEL)
	return count

def return_label(Input_Vector, node, row):
	# GET THE S_ATTR FROM NODE AND FIND THE VALUE IN THE ROW
	label = None

	#if node.s_attr != None:
	testing = row[header.index(node.s_attr)]
	for n in node.node_vector:
		if n.name == testing:
			#print("IN N.NAME = ", n.name)
			if(n.s_attr != None):
				label = return_label(Input_Vector, n, row)
			if label == None:
				label = print_label(Input_Vector, n)

	return label

def return_label_2(Input_Vector, node, row):
    label = None
    
    # If node.s_attr != None:
    testing = row[header.index(node.s_attr)]
    for n in node.node_vector:
        if n.name == testing:
            if(n.s_attr != None):
                label = return_label_2(Input_Vector, n, row)
            if label == None:
                label = n.label
        
    return label

def check_labels(Input_Vector, node, row):
	label = return_label(Input_Vector, node, row)
	#print("IN CL")
	#print("label")	
	#print(row[TEST_LABEL])
	if label == row[TEST_LABEL]:
		return True
	else:
		return False

def check_labels_2(Input_Vector, node, row):
    label = return_label_2(Input_Vector, node, row)
    if label == row[TEST_LABEL]:
        return True
    else:
        return False

def check_accuracy(Input_Vector, node, Test_Vector):
	total = 0
	total_true = 0
	for i in range(len(Test_Vector)):
		if check_labels(Input_Vector, node, Test_Vector[i]):
			total_true += 1
		total += 1
	return float(total_true) / total * 100

def check_accuracy_2(Input_Vector, node, Test_Vector):
    total = 0
    total_true = 0
    for i in range(len(Test_Vector)):
        if check_labels_2(Input_Vector, node, Test_Vector[i]):
            total_true += 1
        total += 1
    return float(total_true) / total * 100

class Node:
	def __init__(self, name):
		# Name of what we hold (Color, Size, Etc)
		self.name = name
		# Name of what we are splitting on next
		self.s_attr = None
		# Let this be a list of integers that hold the indexes of the splits
		self.sp_vector = []
		# Let this be a list of nodes so we know where to go
		self.node_vector = []
		# Let this be a list of all attributes it can currently split on
		self.attr2sp = []
		# LET THIS BE THE LABEL WE USE FOR OUR HYPOTHESIS
		self.label = 0

	def output(self):
		for n in self.node_vector:
			print(n.name)
			print(n.sp_vector)

test2()