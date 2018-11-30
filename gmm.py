import numpy as np
import time

class GMM():

    def __init__(self,n_mixture,n_harmonics=5,weights=None,means=None,sigmas=None,n_iter=100,print_every=1,crit=1e-5,verbose=False):
        if n_harmonics>n_mixture:
            raise ValueError('number of harmonics must be smaller or equal to number of mixtures!')
        self.n_mixture=n_mixture
        self.n_iter=n_iter
        self.print_every=print_every
        self.crit=crit
        self.weights=weights
        self.means=means
        self.sigmas=sigmas
        self.verbose=verbose
        self.n_harmonics=n_harmonics


    def k_mean(self,data,K,rep=1,n_iter=1000,crit=1e-5,centroids=None):
        #data is an (N,D) numpy array
        #returns a list of centroids (np.array), clusters (list of np arrays), summed distance
        N=len(data)#amount of data
        best_model=[]
        #-----run k-means rep times
        for r in range(0,rep):
        
            #----init centroids:----
            centroids=np.zeros(K)#allocate memory
            mean=np.mean(data)
            sigma=np.std(data)
            #initialize centroids
            centroids[:]=np.random.normal(mean,sigma,K)

            #-----k-mean iterations----
            if self.verbose:
                print('start k-mean repetition: '+str(r+1)+'...')
            clusters = [[] for _ in range(K)]
            #E-step: assign points to neaerest centroid
            last_distance=1e10
            new_distance=1e9
            best_model=[centroids,clusters,last_distance]
            iter=0
            while (last_distance-new_distance)/last_distance>crit and iter<n_iter:
                last_distance=new_distance
                new_distance=0
                clusters = [[] for _ in range(K)]
                for n in range(0,N):
                    squared_distance=np.power(data[n]-centroids[:],2)#broadcasting
                    idx_min=np.argmin(squared_distance)
                    clusters[idx_min].append(data[n].tolist())
                    new_distance+=np.power(squared_distance[idx_min],1/2)

                #M-step: move centroids to cluster center
                for k in range(0,K):
                    if clusters[k]:
                        centroids[k]=np.mean(clusters[k])
                iter+=1
                if np.mod(iter,5)==0 and self.verbose:
                    print('iter: '+str(iter))
            if new_distance<best_model[2]:
                best_model=[centroids,clusters,new_distance]

        for k in range(0,K):
            best_model[1][k]=np.asarray(best_model[1][k])

        #check for small clusters
        for k in range(0,K):
            if len(best_model[1][k])<2+int(N/K/10):
                #retrain completely
                if self.verbose:
                    cluster_size=len(best_model[1][k])
                    print('Retrain k-mean because of a too small cluster ('+str(cluster_size)+').')
                best_model=self.k_mean(data,K,rep=rep,n_iter=n_iter)
        return best_model


    def init_params(self,x,art_init=False):
        N=len(x)
        if art_init==True and self.means is None or self.weights is None or self.sigmas is None:
            if self.verbose:
                print('artificial initialization....')
         #note, I will normalize the weights at the end of this function
            self.means=np.zeros(self.n_mixture)
            self.sigmas=np.zeros(self.n_mixture)
            self.weights=np.zeros(self.n_mixture)
            self.normalizations=np.zeros(self.n_mixture)
            base_tone=15
            base_sigma=1
            for i in range(0,self.n_harmonics):
                self.means[i]=(i+1)*base_tone
                self.sigmas[i]=(i/8+1)*base_sigma#sigmas get bigger for larger harmonics (less absolut freq. reso.)
                self.weights[i]=1*(2-i/self.n_harmonics)#weights become smaller for larger harmonics
                self.normalizations[i]=np.sqrt(2*np.pi)/self.sigmas[i]
            self.means[self.n_harmonics:]=np.linspace(25,100,self.n_mixture-self.n_harmonics)#elemnts of [10,120]
            self.sigmas[self.n_harmonics:]=np.random.rand(self.n_mixture-self.n_harmonics)*5+3#eleements of [3,8]
            self.weights[self.n_harmonics:]=np.random.rand(self.n_mixture-self.n_harmonics)+0.1#eleements of [0.1,1.1]
            self.weights/=np.sum(self.weights)
            for i in range(self.n_harmonics,self.n_mixture):
                self.normalizations[i]=1/np.sqrt((2*np.pi))/self.sigmas[i]

        elif self.means is None or self.weights is None or self.sigmas is None:
            #k-mean init:
            if self.verbose:
                print('initialization with kmeans...')
            centroids,clusters,last_distance=self.k_mean(x,self.n_mixture)
            self.means=centroids
            self.weights=np.zeros(self.n_mixture)
            self.sigmas=np.zeros(self.n_mixture)
            self.normalizations=np.zeros(self.n_mixture)
            for m in range(0,self.n_mixture):
                self.weights[m]=len(clusters[m])/N
                #calculate variance of clusters
                variances_m=np.var(clusters[m])
                self.sigmas[m]=np.sqrt(variances_m)
                self.normalizations[m]=1/np.sqrt((2*np.pi))/self.sigmas[m]          
        else:
            if self.verbose:
                print('Using given initial parameters....')
        if np.abs(np.sum(self.weights)-1)>1e-8:
            raise ValueError('weights are not well initialized: '+str(self.weights)+' sum to: '+str(np.sum(self.weights)))

    def fit(self,x,collapse=None,l=0.5,art_init=False):
        #x is a np array of shape (N,D), where N is the # of data points and D the dimension
        #collapse: the covariance matrix feels a force towards a small values if 
        #the squarroot of the determinant is smaller than collapse*2. The force points
        #towards its scaled version with sqrt of determinant equal to collapse
        #l describes the force: 
        #if l=0 nothing happens, if l=1, the determinant of the new covar becomes collapse*2
        
        self.init_params(x,art_init=art_init)
        N=len(x)
        #---EM-algorithm---
        start_time = time.time()
        print_time=self.print_every
        LL=[-1e10]
        LL.append(self.score(x))
        improvement=(LL[-1]-LL[-2])/np.abs(LL[-2])
        iter=0
        weights=np.ones(self.n_harmonics)
        for i in range(self.n_harmonics):
            weights[i]/=(i+1)
        while improvement>self.crit and iter<self.n_iter:
            #----E-step:
            gamma=self.get_responsibilities(x)
            #----M-step:
            s_gamma=np.sum(gamma,axis=0)
            #---weights----
            self.weights=s_gamma/N
            #---means---
            #tie the harmonics together
            old_means=self.means
            new_means=np.einsum('nm,n->m',gamma,x)/s_gamma
            d_mean=new_means-old_means
            d_harmonics=np.average(d_mean[0:self.n_harmonics],weights=weights)
            for i in range(0,self.n_harmonics):
                new_means[i]=old_means[i]+d_harmonics*(i+1)
            #check if still in harmonic range
            if new_means[0]>35:
                print('Warning: The base tone seems unnaturally high: '+str(new_means[0])+' We throw it down')
                base_tone=15
                for i in range(0,self.n_harmonics):
                    self.means[i]=(i+1)*base_tone
            self.means=new_means
            #---sigmas----
            old_sigmas=self.sigmas
            for m in range(0,self.n_mixture):
                x_min_mu_m=(x-self.means[m])
                empirical_vars=np.einsum('n,n->n',x_min_mu_m,x_min_mu_m)
                new_sigma=np.sqrt(np.einsum('n,n',gamma[:,m],empirical_vars)/s_gamma[m])
                if new_sigma<0.1:
                    print('Warning! component: '+str(m+1)+' was in dangeour to collapse. We blowed it up again.')
                    new_sigma=1.5
                self.sigmas[m]=new_sigma
            d_sigma=self.sigmas-old_sigmas
            d_harmonics=np.average(d_sigma[0:self.n_harmonics],weights=weights)
            for i in range(0,self.n_harmonics):
                self.sigmas[i]=old_sigmas[i]+d_harmonics*(i/8+1)
            self.normalizations=1/np.sqrt((2*np.pi))/self.sigmas


            # for m in range(self.n_harmonics,self.n_mixture):
            #     x_min_mu_m=(x-self.means[m])
            #     empirical_vars=np.einsum('n,n->n',x_min_mu_m,x_min_mu_m)
            #     new_sigma=np.sqrt(np.einsum('n,n',gamma[:,m],empirical_vars)/s_gamma[m])
            #     if new_sigma<0.1:
            #         print('Warning! component: '+str(m+1)+' was in dangeour to collapse. We blowed it up again.')
            #         new_sigma=1.5
                # self.sigmas[m]=new_sigma
                # self.normalizations[m]=1/np.sqrt((2*np.pi))/self.sigmas[m]
            LL.append(self.score(x))                
            improvement=(LL[-1]-LL[-2])/np.abs(LL[-2])
            current_time=time.time()-start_time
            if current_time>print_time and self.verbose:
                print('Epoch: '+str(iter)+', Training time: '+str(int(current_time))+'s, Likelihood: '+str(LL[iter+1])+', last improvement: '+str(improvement))
                print_time=current_time+self.print_every

            iter+=1

        return LL

    def get_responsibilities(self,x):
        #x is the full data set of shape (N,D)
        N=len(x)
        gamma=np.zeros((N,self.n_mixture))
        for n in range(0,N):
            for m in range(0,self.n_mixture):
                gamma[n,m]=self.weights[m]*self.component_density(m,x[n])
            denominator=np.sum(gamma[n,:])
            gamma[n,:]=gamma[n,:]/denominator
        return gamma


    def sample(self,n,truncate=None,covar_bias=1):
        #truncate is an option to forbit sampling from the tails of the Gaussians.
        #truncate is the amount of standarddeviations we want to sample from
        #if truncate goes to infinity, this corresponds to truncate=None
        #covar_bias scales the covariance matrix for the sampling by the factor covar_bias**2
        #you must choose between truncate and covar_bias. If both are not None, its truncate over covar_bias
        sample=np.zeros(n)
        mixture_idx = np.linspace(0,self.n_mixture-1,self.n_mixture).astype(int)
        weights = self.weights
        components=np.random.choice(mixture_idx, n, p=weights)
        # if truncate is not None:
        #     i=0
        #     if self.D==1:
        #         while i<n:
        #             sample_point=np.random.normal(self.means[components[i]],np.sum(self.covars[components[i]]),1)
        #             if (sample_point-self.means[components[i]])**2/np.sum(self.covars[components[i]])<=truncate**2:
        #                 sample[i,:]=sample_point
        #                 i+=1

        #     while i<n:
        #         sample_point=np.random.multivariate_normal(self.means[components[i]],self.covars[components[i]],1)
        #         if np.einsum('i,i',sample_point-self.means[components[i]],np.einsum('ij,j', self.invcovars[components[i]], sample_point-self.means[components[i]]))<=truncate**2:
        #             sample[i,:]=sample_point
        #             i+=1
        for i in range(n):
            sample[i]=np.random.normal(self.means[components[i]],self.sigmas[components[i]],1)
        return sample

    def mixture_densitiy(self,x):
        p=0
        for m in range(0,self.n_mixture):
            p+=self.weights[m]*self.component_density(m,x)
        return p

    def score(self,x):
        LL=0
        for i in range(0,len(x)):
            LL+=np.log(self.mixture_densitiy(x[i]))
        return LL
               

    def component_density(self,component,x):
        return self.normalizations[component]*np.exp(-(x-self.means[component])**2/2/self.sigmas[component]**2)

