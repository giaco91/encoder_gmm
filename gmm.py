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
            self.base_tone=15
            self.base_sigma=1
            for i in range(0,self.n_harmonics):
                self.means[i]=(i+1)*self.base_tone
                self.sigmas[i]=(i/8+1)*self.base_sigma#sigmas get bigger for larger harmonics (less absolut freq. reso.)
                self.weights[i]=1*(2-i/self.n_harmonics)#weights become smaller for larger harmonics
                self.normalizations[i]=np.sqrt(2*np.pi)/self.sigmas[i]
            # self.means[self.n_harmonics:]=np.linspace(self.base_tone*(self.n_harmonics+1),110,self.n_mixture-self.n_harmonics)
            # for j in range(self.n_harmonics,self.n_mixture):
            #     self.means[j]=self.base_tone*(1/2+(self.n_harmonics-j+self.n_harmonics))
            self.check_additional_gaussians(init=True)
            self.sigmas[self.n_harmonics:]=3
            self.weights[self.n_harmonics:]=0.5
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

        # self.sigmas[-1]=4    
        # print(self.means)
        # print(self.sigmas)
        # print(self.weights)
        # means=np.copy(self.means)
        # sigmas=np.copy(self.sigmas)
        # weights=np.copy(self.weights)
        # self.means[-1]=means[-2]
        # self.means[-2]=means[-1]
        # self.sigmas[-1]=sigmas[-2]
        # self.sigmas[-2]=sigmas[-1]
        # self.weights[-1]=weights[-2]
        # self.weights[-2]=weights[-1]
        # print(self.means)
        # print(self.sigmas)
        # print(self.weights)
        # self.check_additional_gaussians()
        # print(self.means)
        # print(self.sigmas)
        # print(self.weights)
        if np.abs(np.sum(self.weights)-1)>1e-8:
            raise ValueError('weights are not well initialized: '+str(self.weights)+' sum to: '+str(np.sum(self.weights)))

    def check_additional_gaussians(self,init=False):
        #this is written for 4 hamronics and 3 additional Gaussians
        #the first is between harmonic component 2 and 3
        #the second between harmonic compontn 3 and 4
        #and the last two above the hightest hamronics
        reinit=False
        if init==True:
            if self.n_harmonics!=4 or self.n_mixture!=8:
                raise ValueError('the number of harmonics and additional gaussians must be 4')
            self.means[4]=(self.means[1]+self.means[2])/2
            self.means[5]=(self.means[2]+self.means[3])/2
            self.means[6]=self.means[3]+20
            self.means[7]=self.means[3]+40
        else:
            # #if the order is not correct, just reinitialize them
            # if not self.means[1]<self.means[4]<self.means[2] or abs(self.means[4]-self.means[1])<1 or abs(self.means[4]-self.means[2])<1:
            #     self.means[4]=(self.means[1]+self.means[2])/2
            #     reinit=True
            # if not self.means[2]<self.means[5]<self.means[3] or abs(self.means[5]-self.means[2])<1 or abs(self.means[5]-self.means[3])<1:
            #     self.means[5]=(self.means[2]+self.means[3])/2
            #     reinit=True
            # if self.means[6]<self.means[3] or abs(self.means[6]-self.means[3])<1:
            #     self.means[6]=self.means[3]+10
            #     reinit=True
            # if self.means[7]<self.means[6]:
            #     self.means[7]=self.means[6]+10
            #     reinit=True
            #-----
            #if the order is not correct, just reinitialize them
            # if not self.means[1]<self.means[4]<self.means[2]:
            #     self.means[4]=(self.means[1]+self.means[2])/2
            #     reinit=True
            # if not self.means[2]<self.means[5]<self.means[3]:
            #     self.means[5]=(self.means[2]+self.means[3])/2
            #     reinit=True
            # if self.means[6]<self.means[3]:
            #     self.means[6]=self.means[3]+10
            #     reinit=True
            # if self.means[7]<self.means[6]:
            #     self.means[7]=self.means[6]+10
            #     reinit=True
            # if reinit==True:
            #-------
            #just keep track of the order
            if not self.means[4]<self.means[5]<self.means[6]<self.means[7]:
                sorted_idx=np.argsort(self.means[4:])
                self.means[4:]=np.sort(self.means[4:])
                unsorted_sigmas=np.copy(self.sigmas[4:])
                unsorted_weights=np.copy(self.weights[4:])
                self.sigmas[4:]=unsorted_sigmas[sorted_idx]
                self.weights[4:]=unsorted_weights[sorted_idx]
                print('resorted the additional gaussians')

        # if reinit==True:
        #     print('reinit: '+str(self.means))
        return reinit




    def fit(self,x,collapse=None,l=0.5,art_init=False,pull=0,tie_harmonic_var=True):
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
        init_score=self.score(x)
        if init_score<-10000:
            self.init_params(x,art_init=True)
            print('reinitializing parameter because of low initial score.')
        LL.append(init_score)
        improvement=(LL[-1]-LL[-2])/np.abs(LL[-2])
        iter=0
        weights=np.ones(self.n_harmonics)
        pull_nonharmonics=0.1
        pull_sigma=0.5
        for i in range(self.n_harmonics):
            weights[i]/=(i+1)
        reinit=False#if we need to reinizialize some parameters, we need to correct the improvement
        while improvement>self.crit and iter<self.n_iter:
            #----E-step:
            gamma=self.get_responsibilities(x)
            #----M-step:
            s_gamma=np.sum(gamma,axis=0)
            #---weights-----------
            self.weights=s_gamma/N
            #---means-------
            #tie the harmonics together
            new_means=np.einsum('nm,n->m',gamma,x)/s_gamma
            d_mean=new_means-self.means
            d_harmonics=np.average(d_mean[0:self.n_harmonics],weights=weights)
            for i in range(0,self.n_harmonics):
                new_means[i]=self.means[i]+d_harmonics*(i+1)
            #bias the non_harmonics to be between the harmonics
            for j in range(self.n_harmonics,self.n_mixture):
                new_means[j]=pull_nonharmonics*(j*20-40)+(1-pull_nonharmonics)*(self.means[j]+d_mean[j])
            #check if still in harmonic range
            if new_means[0]>40 or new_means[0]<4:
                print('Warning: The base tone seems unnaturally high or low: '+str(new_means[0])+' We throw it back to 15')
                for i in range(0,self.n_harmonics):
                    new_means[i]=(i+1)*self.base_tone
                reinit=True
            self.means=new_means
            self.check_additional_gaussians()
            #---sigmas----
            old_sigmas=np.copy(self.sigmas)
            for m in range(0,self.n_mixture):
                x_min_mu_m=(x-self.means[m])
                empirical_vars=np.einsum('n,n->n',x_min_mu_m,x_min_mu_m)
                new_sigma=np.sqrt(np.einsum('n,n',gamma[:,m],empirical_vars)/s_gamma[m])
                self.sigmas[m]=new_sigma
            d_sigma=self.sigmas-old_sigmas

            if tie_harmonic_var==True:
                d_harmonics=np.average(d_sigma[0:self.n_harmonics],weights=weights)
                for i in range(0,self.n_harmonics):
                    self.sigmas[i]=pull*((i/8+1)*self.base_sigma)+(1-pull)*(old_sigmas[i]+d_harmonics*(i/8+1))
            else:
                for i in range(0,self.n_harmonics):
                    self.sigmas[i]=pull*((i/8+1)*self.base_sigma)+(1-pull)*(old_sigmas[i]+d_sigma[i])
            for j in range(self.n_harmonics,self.n_mixture):
                self.sigmas[j]=pull_sigma*2*(j-self.n_harmonics+1)/2+(1-pull_sigma)*(self.sigmas[j])
            for i in range(self.n_mixture):
                if self.sigmas[i]<0.1:
                    print('Warning! component: '+str(m+1)+' was in dangeour to collapse. We blowed it up again.')
                    self.sigmas[i]=1.5
                    reinit=True
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
            LL.append(self.score(x))   #likelihood with inertia 
            if reinit==False:            
                improvement=(LL[-1]-LL[-2])/np.abs(LL[-2])
            else: 
                improvement=2*self.crit #improvement such that the while loop does not stop
                reinit=False
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
            if self.sigmas[components[i]]<0.01:
                print('the sigma in component '+str(components[i])+' is too small: '+str(self.sigmas[components[i]]))
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

