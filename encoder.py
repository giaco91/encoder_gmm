import numpy as np
import matplotlib.pyplot as plt
import sys
from gmm import GMM
sys.path.append('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/codes_masterthesis')
from utils import *
from pomegranate import *


image_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/data_masterthesis/b7r16/images/day00_b7r16/images'


def transform(im):
    im = from_polar(im)
    im,phase = lc.magphase(im)
    im = np.log1p(im)
    return im

def samples_to_vector(samples,a,vector_dim=129):
	n_samples=len(samples)
	vector=np.zeros(vector_dim)
	for n in range(n_samples):
		idx=int(np.floor(samples[n]))
		if idx>=0 and idx<=vector_dim-1:
			vector[idx]+=1
	vector/=np.sum(vector)
	vector*=a
	return vector

def get_vector(model,a,vector_dim,n=2000):
	samples=model.sample(n)
	return samples_to_vector(samples,a,vector_dim)

def get_base_freq(means):
	rounded_means=np.round(means)
	max_mean=np.max(means)
	n_means=len(means)
	score=(0,0)
	crit=1
	for m in range(0,n_means):
		s=0
		h=0
		zero_freq=means[m]
		while h*means[m] < max_mean+1:
			if np.round(h*means[m]) in list(rounded_means):
				s+=1
			h+=1
		if s>score[0]:
			score=(s,m)

	return means[m]

def get_samples(vector,n=2000):
	#vector should have only positive components
	a=np.sum(vector)
	eps=1e-5
	new_vector=np.copy(vector)+eps #to make sure we have no absolute zero vectors
	 #we store the probability amplitude of the vector
	new_vector/=np.sum(new_vector) #we normalize vector to have equally many sample for every vector
	n_samples=np.round(new_vector*n).astype(int) #contains the number of samples per frequency bin
	n=np.sum(n_samples) #we got some rounding errors
	samples=np.zeros(n)#here we store the sampels
	pointer=0
	for f in range(0,len(vector)):
		samples[pointer:pointer+n_samples[f]]=np.random.random(n_samples[f])+f
		pointer+=n_samples[f]
	return samples,a
	
# def init_with_GMM(n_mixtures,X):
#     max_iterations=400
#     X=concat(X)
#     print('training GMM...')
#     GMM = GeneralMixtureModel.from_samples(MultivariateGaussianDistribution, n_components=n_mixtures, X=X,verbose=True, max_iterations=max_iterations)
#     #s=[]
#     dist=[]
#     for k in range(n_mixtures):
#         mean=GMM.distributions[k].parameters[0]
#         cov=GMM.distributions[k].parameters[1]
#         dist.append(MultivariateGaussianDistribution(mean,cov))

#     trans_mat=np.ones((n_mixtures,n_mixtures))/n_mixtures
#     starts=np.ones(n_mixtures)/n_mixtures
#     hmm=HiddenMarkovModel.from_matrix(trans_mat,dist,starts)
#     return hmm


#load image
STFT=np.load(image_path+'/54787.npy')
spec=transform(STFT)[:,:]
rec_spec=np.zeros((spec.shape))
max_iterations=100
n_mixtures=7
n_harmonics=4
verbose=False
gmm=GMM(n_mixtures,n_harmonics=n_harmonics,n_iter=100,print_every=0.5,crit=1e-4,verbose=verbose)
for i in range(0,spec.shape[1]):
	print('encode vector: '+str(i)+'...')
	vec_orig=spec[:,i]
	samples,a=get_samples(vec_orig)
	#gmm.fit(samples,collapse=2,l=0.9)
	gmm.fit(samples,art_init=True)
	rec_spec[:,i]=get_vector(gmm,a,spec.shape[0])
	# print(gmm.means)
	# print(gmm.sigmas)

plt.imshow(spec,origin='lower')
plt.show()
plt.imshow(rec_spec,origin='lower')
plt.show()

# vec_orig=transform(STFT)[:,89]
# samples,a=get_samples(vec_orig)
# gmm.fit(samples,art_init=True)
# print('------------')
# vec_orig=transform(STFT)[:,90]
# samples,a=get_samples(vec_orig)
# gmm.fit(samples,art_init=True)
# vector=get_vector(gmm,a,spec.shape[0])
# print(gmm.means)
# print(gmm.sigmas)
# print(gmm.weights)
# plt.plot(vector,'r-')
# plt.plot(vec_orig,'b-')
# plt.plot(gmm.means,np.zeros(len(gmm.means)),'b.',markersize=1.5,color='red')
# plt.show()




