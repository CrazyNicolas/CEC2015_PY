# CEC2015_PY
The project is a Python implement based on the problem definition and [C, Java and Matlab implement](https://github.com/P-N-Suganthan/CEC2015-Learning-Based) of cec2015 test functions. These functions include basic functions:

**Sphere, Ellipsoidal,  Bent_cigar, Discus, Dif_powers, Rosenbrock, Ackley, Weierstrass, Griewank, Rastrigin, Schwefel,  Katsuura, Grie_rosen, Escaffer6, Happycat, Hgbat.**

And **Composition** and **Hybrid** functions base on basic functions.

## cec15_test_func.py

### 1. Basic Functions
#### 1) Code Structure: 
The **Basic Function** inherits the **Problem** class, inherits the problem data (shift, rotate matrix) generation methods and read-write file methods of the parent class, and overloads the func evaluatiton method.
#### 2) Methods: 
**<1> Construction Method**: The constructor parameters of the basic problem are specified as [problem dimension (int), offset (shift, numpy one-dimensional vector), rotation matrix (numpy two-dimensional matrix)].

**<2> Evaluation Method (func)**: The specific problem class overloads the evaluation method of parent class. The parameter is [single solution vector (numpy one-dimensional vector, no dimension validity check is provided)] and returns the evaluation value.

**<3> Read Method (read)**: Given the file path, problem type and number of problem set instances to be read, the read data returns a batch of problem instance objects.

**<4> Generator Method (generator)**: Given the problem type (string, one of the function names listed above), the problem dimension (int) and the generated quantity, the method will randomly generate the offset and rotation matrix and return the specified problem data [type, dimension, offset, rotation].

**<5> Rotate Matrix Generator (rotate_gen)**: Randomly generate a rotation matrix for a given dimension (int).

**<6> Instance Storing Method (store_instance)**: Given [problem data, file path (string)], the method saves the problem data.

**<7> Get Method (get_instance)**: After a batch of problem data is given, a batch of problem instance objects are returned.

### 2. Hybrid Functions
#### 1) Code Structure: 
The **Hybrid Function** is realized by calling the basic problem instance objects.
#### 2) Methods: 

**<1> Construction Method**: The parameter is the data required for instantiation, including total dimension, number of sub problems, length vector of sub problems, reordering vector and list of problem instance objects.

**<2> Evaluation Method (func)**: The parameter is a solution vector (numpy one-dimensional vector, which does not provide dimension validity check), and the evaluation value is calculated according to the rules described in the problem definition document.

**<3> Generator Method (generator)**:  Given the storage file path (**None** if you choose not to save), specify the total dimension, number of sub problems, list of sub problem types, number of generated problems, whether to save and align. When the dimension and number of questions are **less than or equal to 0**, or the question list is **None or empty**, it will be generated randomly. Various parameters such as subproblem length division are randomly generated. The subproblem is randomly selected from the candidate problem list and initialized randomly (using the basic function's own generating method). Returns a batch of mixed problem data in a list.

**Note**: the alignment is because the Dataloader will stack the problem data, and the different lengths of sub problems will lead to the stack Failed, so the sub problem data is supplemented with 0 to reach the same dimension to be compatible with the Dataloader. Stored in file The data in is unaligned raw data.

**<4> Read Method (read)**: Given the file path, the number of problem instances to be read and whether they are aligned or not. Method reads and returns file data.

**<5> Get Method (get_instance)**: After a batch of problem data is given, a batch of problem instance objects are returned.

### 3. Composition Functions
#### 1) Code Structure: 
The **Composition Function** is realized by calling the basic problem instance objects.
#### 2) Methods: 

**<1> Construction Method**: The parameter is the data required for instantiation, including dimension, number of sub problems, bias, etc. and the list of problem instance objects.

**<2> Evaluation Method (func)**: parameter is a solution vector (numpy one-dimensional vector, which does not provide dimension validity check), and the evaluation value is calculated according to the rules described in the problem definition document.

**<3> Generator Method (generator)**: Similar to the **Hybrid Function**, given the storage file path (**None** if you choose not to save), specify the total dimension, number of sub problems, list of sub problem types, number of generated problems and whether to save or not. When the dimension and number of questions are **less than or equal to 0**, or the question list is **None or empty**, it will be generated randomly. The parameters used in function evaluation, such as bias and F, are randomly generated. The subproblem is randomly selected from the candidate problem list and initialized randomly (using the basic function's own generating method). Return a batch of composition problem data in list form.

**<4> Read Method (read)**: Given the file path, the number of problem instances to be read. Method reads and returns file data.

**<5> Get Method (get_instance)**: After a batch of problem data is given, a batch of problem instance objects are returned.

## cec15_test_func.py

### 1. Dataset Class: Tester

#### 1) Code Structure: 
It inherits the **Dataset** class of **torch.utils.data**, contains the generation and reading interface of problem dataset, and is designed to adapt to the data format of **Dataloader**. The data member of the class is a list of problem data, which contains all the data in the dataset.
#### 2) Methods:

**<1> Construction Method**: Give the parameters required for dataset initialization, including file path, problem type, problem dimension, sampling quantity and offset. When the file path is **None**, the data set is randomly generated and not saved as a file. If it is not **None**, the data is read from the file.

**<2> Dataset generation**: For a given file path (**None* if you choose not to save), generated quantity (size), problem type, dimension, number of sub problems, list of sub problem types, and whether to save or not. The data set is generated by calling the generation function of the corresponding problem, and the parameters not required in the generation will be ignored (for example, the parameters related to the sub problem will be ignored when generating the basic problem data set). The saved file name is '[problem type]_ D [dimension].txt'.

**<3> Data set reading**: Given the file path, problem type, read quantity and offset, the problem data is returned.
#### 3) Call: 
Instantiate the **Tester** class object of the corresponding problem as needed, and get the data in batch after entering **Dataloader**. The obtained data uses the **get_instance** method of the corresponding problem type to obtain the instance object list. The problem class can be obtained from the **problem_types** dictionary with the problem name string.

