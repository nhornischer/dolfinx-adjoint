import numpy as np
def gradient_normalisation(gradients : np.ndarray, bounds : np.ndarray, interval = np.array([-1., 1.])):
    """Normalizes the gradients of a sample

    Normalizes the gradients of a sample from the interval using the bounds

    Args:
        gradients (np.ndarray): The gradients to be normalized. Has either shape (M, m, d) or (m, d).
        bounds (np.ndarray): The bounds used for normalization. Has shape (m, 2). 
        interval (np.ndarray): Scaling interval. Default [-1., 1.].

    Returns:
        np.ndarray: The normalized gradients
    """

    print(f"Normalizing gradients shape({np.shape(gradients)}) with bounds shape({np.shape(bounds)}) and interval shape({np.shape(interval)})")
    print(f"Original gradients: \n{gradients} with bounds \n{bounds}")

    # Normalize
    gradients = gradients * (bounds[:, 1] - bounds[:, 0]) / (interval[1] - interval[0])

    print(f"Normalized gradients: \n{gradients}")

    return gradients
def normalizer(sample: np.ndarray, bounds: np.ndarray or None = None,
               interval: np.ndarray = np.array([-1., 1.])):
    """Normalizes a sample
    
    Normalizes a sample to the interval [-1, 1] using the bounds
    
    Args:
        sample (np.ndarray): The sample to be normalized. Has either shape (M, m) or (m,).
        bounds (np.ndarray): The bounds used for normalization. Has shape (m, 2).
        interval (np.ndarray): The interval for normalization. Default [-1., 1.]
    Returns:
        np.ndarray: The normalized sample
    """

    print(f"Normalizing sample shape({np.shape(sample)}) with bounds shape({np.shape(bounds)}) and interval shape({np.shape(interval)})")
    print(f"Original sample: \n{sample} with bounds \n{bounds}")

    assert len(sample.shape) <= 2, "Sample has more than 2 dimensions"

    # Make sample 2D if it is 1D
    if len(sample.shape) == 1:
        sample = sample.reshape(1, -1)

    if bounds is None:
        bounds = np.min(sample, axis=0).reshape(1, -1)
        bounds = np.concatenate((bounds, np.max(sample, axis=0).reshape(1, -1)), axis=0).T

    # Check if bounds are valid
    assert bounds.shape[0] == sample.shape[1] and bounds.shape[1] == 2, f"Bounds do not match sample dimensions. Have shape {np.shape(bounds)} but should have shape ({sample.shape[1]}, 2)"
    assert np.all(bounds[:, 0] < bounds[:, 1]), "Lower bounds musst be smaller than Upper bounds"

    # Make interval 2D if it is 1D
    if len(interval.shape) == 1:
        interval = interval.reshape(1, -1)
    assert interval.shape[0] == 1 or interval.shape[0] == bounds.shape[0], f"Interval shape {interval.shape} doesn't match the bounds shape {bounds.shape}"

    sample = (sample - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0]) * (interval[:, 1] - interval[:, 0]) + interval[:, 0]

    # Make sample 1D if it is unnecessary 2D
    if sample.shape[0] == 1:
        sample = sample.reshape(-1)

    print(f"Normalized sample: \n{sample}")
    return sample

class ASFEniCSx:
    def __init__(self, M, m,k,  function, samples, bounds = None):
        self.M = M
        self.m = m
        self.k = k
        self.function = function
        self.bounds = bounds
        self.samples = samples

    def evaluate_gradients(self, **kwargs):
        """Evaluates the gradients of the function at the samples

        Args:
            
        Returns:
            np.ndarray: Matrix containing the gradients of the function at the samples in the rows
        """

        # Check if additional arguments are given
        gradients = np.zeros([self.M, self.m])
        print("Evaluating gradients for active subspace construction")
        for i in range(self.M):
            print(f"Sample {i+1}/{self.M}", end="\r")
            gradients[i] = self.function(self.samples[i,:], **kwargs)[1]
        self.gradients = gradients

        # Normalize the gradients according to the chain rule with the bounds from the sampling space to the range [-1, 1]
        if self.bounds is not None:
            print("Normalizing gradients")
            for i in range(self.M):
                gradients[i,:] = gradient_normalisation(gradients[i,:], self.bounds)
        else:
            print("No bounds found in the sampling object. Gradients are not normalized.")
        return gradients

    def covariance(self, gradients : np.ndarray):
        """Approximates the covariance matrix of the gradient of the function

        The calculation of the gradient is defined directly in the functional.
        The covariance matrix is approximated by the outer product of the gradient.
        
        Args:
            gradients (numpy.ndarray): Matrix containing the gradients of the function at the samples in the rows

        Returns:
            np.ndarray: Approximated covariance matrix with dimensions m x m    
        """
        weights = np.ones((self.M, 1))/self.M

        covariance = np.dot(gradients.T, gradients * weights)
        return covariance

    def estimation(self):
        """Calculates the active subspace using the random sampling algorithm of Constantine et al.
        Corresponds to Algorithm 3.1 in the book of Constantine et al.

        Args:
        
        Returns:
            np.ndarray: Matrix of eigenvectors stored in the columns
            np.ndarray: Vector of eigenvalues
        """

        # Evaluate the gradients of the function at the samples
        print("Constructing the active subspace using the random sampling algorithm")
        if not hasattr(self, 'gradients'):
            self.evaluate_gradients()
        else:
            print("Gradients are already evaluated, skipping evaluation. Make sure the gradients are up to date.")

        # Construct the covariance matrix
        convariance_matrix = self.covariance(self.gradients)

        # Calculate the eigenvalues and eigenvectors of the covariance matrix
        S, U = self.calculate_eigenpairs(convariance_matrix)

        self._eigenvalues = S
        self._eigenvectors = U

        print(f"Active subspace constructed")

        return (self._eigenvectors, self._eigenvalues)

    def partition(self, n : int):
        """Partitions the active subspace into two subspaces of dimension n and m-n

        Args:
            n (int): Dimension of the active subspace

        Returns:
            np.ndarray: Matrix containing the active subspace of dimension n
            np.ndarray: Matrix containing the inactive subspace of dimension m-n
        """
        # Check if the eigenvalues are already calculated
        if not hasattr(self, 'eigenvalues'):
            raise("Eigenvalues not calculated yet. Run the random sampling algorithm first.")

        # Check if the dimension of the active subspace is smaller than the dimension of the parameter space
        if n > self.m:
            raise("Dimension of the active subspace must be smaller than the dimension of the parameter space.")

        W1 = self._eigenvectors[:,:n]
        W2 = self._eigenvectors[:,n:]
        self.W1 = W1
        return (W1, W2)

    def bootstrap(self, M_boot : int):
        """ Compute the bootstrap values for the eigenvalues
        
        Args:
            M_boot (int): Number of bootstrap samples
            
        Returns:
            np.ndarray: Bootstrap lower and upper bounds for the eigenvalues
            np.ndarray: Bootstrap lower and upper bounds for the subspace distances
        """
        if not hasattr(self, 'gradients'):
            self.evaluate_gradients()

        if not hasattr(self, 'eigenvalues'):
            self.estimation()

        # Loop over the number of bootstrap samples
        eigenvalues = np.zeros([self.m, M_boot])
        subspace_distances = np.zeros([self.m-1, M_boot])
        for i in range(M_boot):
            print(f"Bootstrap sample {i+1}/{M_boot}", end="\r")
            # Construct bootstrap replicate
            bootstrap_indices = np.random.randint(0, self.M, size = self.M)
            bootstrap_replicate = self.gradients[bootstrap_indices,:].copy()

            # Compute the bootstraped singular value decomposition
            S, U = self.calculate_eigenpairs(self.covariance(bootstrap_replicate))

            for j in range(self.m-1):
                subspace_distances[j,i] = np.linalg.norm(np.dot(self._eigenvectors[:,:j+1].T, U[:,j+1:]), ord=2)
            eigenvalues[:,i] = S
        sub_max = np.amax(subspace_distances, axis=1)
        sub_min = np.amin(subspace_distances, axis=1)
        sub_mean = np.mean(subspace_distances, axis=1)

        # Compute the max and min of the eigenvalues over all bootstrap samples
        e_max = np.amax(eigenvalues, axis=1)
        e_min = np.amin(eigenvalues, axis=1)

        self.e_boot = [e_max, e_min]
        self.sub_boot = [sub_max, sub_min, sub_mean]

        print(f"Bootstrap values calculated")

        return [e_max, e_min], [sub_max, sub_min, sub_mean]

    def calculate_eigenpairs(self, matrix : np.ndarray):
        """Calculates the eigenvalues and eigenvectors of a matrix

        Args:
            matrix (np.ndarray): Matrix to calculate the eigenvalues and eigenvectors of

        Returns:
            np.ndarray: Vector of eigenvalues
            np.ndarray: Matrix of eigenvectors stored in the columns
        """
        e, W = np.linalg.eigh(matrix)
        e = abs(e)
        idx = np.argsort(e)[::-1]
        e = e[idx]
        W = W[:,idx]
        normalization = np.sign(W[0,:])
        normalization[normalization == 0] = 1
        W = W * normalization
        return e, W

    def plot_eigenvalues(self, filename = "eigenvalues.png", true_eigenvalues = None, ylim=None):
        """Plots the eigenvalues of the covariance matrix on a logarithmic scale

        Args:
            filename (str, optional): Filename of the plot. Defaults to "eigenvalues.png".
            true_eigenvalues (np.ndarray, optional): True eigenvalues of the covariance matrix. Defaults to None.
        Raises:
            ValueError: If the covariance matrix is not defined
        """
        if not hasattr(self, "_eigenvectors"):
            raise ValueError("Eigendecomposition of the covariance matrix is not defined. Calculate it first.")
        import matplotlib.pyplot as plt
        fig = plt.figure(filename)
        ax = fig.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if true_eigenvalues is not None:
            ax.plot(range(1, self.k+1), true_eigenvalues[:self.k], marker="o", fillstyle="none", label="True")
        ax.plot(range(1, self.k+1), self._eigenvalues[:self.k], marker="x", fillstyle="none", label="Est")
        if hasattr(self, "e_boot"):
            print("Plotting bootstrap bounds for eigenvalues")
            ax.fill_between(range(1, self.k+1), self.e_boot[0][:self.k], self.e_boot[1][:self.k], alpha=0.5, label = "BI")
        plt.yscale("log")
        plt.xlabel("Index")
        plt.ylabel("Eigenvalue")
        plt.legend()
        plt.grid()
        if ylim is not None:
            plt.ylim(ylim)
        plt.savefig(filename)
        plt.close()

    def plot_subspace(self, filename = "subspace", true_subspace = None, ylim=None):
        """Plots the subspace distances

        Args:
            filename (str, optional): Filename of the plot. Defaults to "subspace.png".
        Raises:
            ValueError: If the covariance matrix is not defined
        """
        if not hasattr(self, "_eigenvectors"):
            raise ValueError("Eigendecomposition of the covariance matrix is not defined. Calculate it first.")
        import matplotlib.pyplot as plt
        fig = plt.figure(filename)
        ax = fig.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if self.k == self.m:
            k = self.m-1
        else:
            k = self.k
        if true_subspace is not None:
            ax.plot(range(1, k+1), true_subspace[:k], marker="o", fillstyle="none", label="True")
        ax.plot(range(1, k+1), self.sub_boot[2][:k], marker="x", fillstyle="none", label="Est")
        if hasattr(self, "sub_boot"):
            print("Plotting bootstrap bounds for subspace distances")
            ax.fill_between(range(1, k+1), self.sub_boot[0][:k], self.sub_boot[1][:k], alpha=0.5, label = "BI")
        plt.xlabel("Subspace Dimension")
        plt.yscale("log")
        plt.ylabel("Subspace Error")
        plt.legend()
        plt.grid()
        if ylim is not None:
            plt.ylim(ylim)
        plt.savefig(filename)
        plt.close()

    def plot_sufficient_summary(self, filename = "sufficient_summary"):

        # TODO: This looks strange
        # Compute the active variable values
        if not hasattr(self, "W1"):
            raise("The active subspace is not defined. If the eigenpairs of the covariance matrix are already calculated, call partition() first.")
        
        active_variable_values = normalizer(self.samples,self.bounds).dot(self.W1)
        values = np.asarray([self.function(self.samples[i,:])[0] for i in range(self.M)])

        n = active_variable_values.shape[1]
        import matplotlib.pyplot as plt

        for i in range(min(n, 2)):
            if n > 1:
                fig = plt.figure(filename + f"univariate_{i+1}.pdf")
            else:
                fig = plt.figure(filename + f"univariate.pdf")
            ax = fig.gca()
            ax.scatter(active_variable_values[:,i], values)
            if n > 1:
                plt.xlabel(f"Active Variable {i+1}")
            else:
                plt.xlabel("Active Variable")
            plt.ylabel("Function Value")
            plt.grid()
            if n > 1:
                plt.savefig(filename + f"univariate_{i+1}.pdf")
            else:
                plt.savefig(filename + f"univariate.pdf")
            plt.close()
        
        if n > 1 and n<=2:
            plt.figure(filename + f"bivariate")
            plt.axes().set_aspect('equal')
            plt.scatter(active_variable_values[:,0], active_variable_values[:,1], c=values, vmin=np.min(values), vmax=np.max(values) )
            plt.xlabel("Active Variable 1")
            plt.ylabel("Active Variable 2")
            ymin = 1.1*np.min([np.min(active_variable_values[:,0]) ,np.min( active_variable_values[:,1])])
            ymax = 1.1*np.max([np.max(active_variable_values[:,0]) ,np.max( active_variable_values[:,1])])
            plt.axis([ymin, ymax, ymin, ymax])    
            plt.grid()
            
            plt.colorbar()
            plt.savefig(filename + f"bivariate.pdf")

            plt.close()

    def plot_eigenvectors(self, filename = "eigenvectors.png", true_eigenvectors = None, n = None, x_ticks = None):
        """Plots the eigenvectors of the covariance matrix

        Args:
            filename (str, optional): Filename of the plot. Defaults to "eigenvectors.png".
            true_eigenvectors (np.ndarray, optional): True eigenvectors of the covariance matrix. Defaults to None.
        Raises:
            ValueError: If the covariance matrix is not defined
        """
        if n is None:
            n = self.k
        if not hasattr(self, "_eigenvectors"):
            raise ValueError("Eigendecomposition of the covariance matrix is not defined. Calculate it first.")
        import matplotlib.pyplot as plt
        fig = plt.figure(filename)
        ax = fig.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        for i in range(n):
            if true_eigenvectors is not None:
                ax.plot(range(1, self.k+1), true_eigenvectors[:,i], marker="o", fillstyle="none", label=f"True ({i+1}))")
            ax.plot(range(1, self.k+1), self._eigenvectors[:,i], marker="x", fillstyle="none", label=f"Est ({i+1})")
        plt.xlabel("Parameter")
        m = np.shape(self._eigenvectors)[0]
        if x_ticks is not None:
            plt.xticks(range(1, m+1), x_ticks)
        plt.ylim([-1,1])
        plt.ylabel("Eigenvector")
        plt.legend()
        plt.grid()
        plt.savefig(filename)
        plt.close()

    # TODO: Check if private/protected variales are returned as objects or as copys.