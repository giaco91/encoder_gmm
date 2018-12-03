import numpy as np
import matplotlib.pyplot as plt
import sys
from gmm import GMM
sys.path.append('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/codes_masterthesis')
from utils import *
from pomegranate import *


image_path='/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/data_masterthesis/b7r16/images/day30_b7r16/images'


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





def get_samples(vector,n=2000):
	eps=1e-5
	new_vector=np.copy(vector)+eps #to make sure we have no absolute zero vectors
	 #we store the probability amplitude of the vector
	power_crit=np.sum(new_vector)/(9*129)
	j=0
	for i in range(len(vector)):
		if new_vector[i]<power_crit:
			new_vector[i]=0
			j+=1
	a=np.sum(vector)
	new_vector/=a #we normalize vector to have equally many sample for every vector
	#filter noise (low power frequency bins):
	#print('number of noisy frequency bins: '+str(j))
	n_samples=np.round(new_vector*n).astype(int) #contains the number of samples per frequency bin
	n=np.sum(n_samples) #we got some rounding errors
	samples=np.zeros(n)#here we store the sampels
	pointer=0

	for f in range(0,len(vector)):
		samples[pointer:pointer+n_samples[f]]=np.random.random(n_samples[f])+f
		pointer+=n_samples[f]
	return samples,a
	

#load image
for j in range(0,1):
	STFT=np.load(image_path+'/9025'+str(j)+'.npy')
	spec=transform(STFT)[:,0:200]
	rec_spec=np.zeros((spec.shape))
	max_iterations=100
	n_mixtures=8
	n_harmonics=4
	verbose=False
	gmm=GMM(n_mixtures,n_harmonics=n_harmonics,n_iter=100,print_every=0.5,crit=1e-5,verbose=verbose)
	# for i in range(0,spec.shape[1]):
	# 	print('encode vector: '+str(i)+'...')
	# 	vec_orig=spec[:,i]
	# 	samples,a=get_samples(vec_orig)
	# 	# rec_spec[:,i]=samples_to_vector(samples,a,vector_dim=129)
	# 	LL=gmm.fit(samples,art_init=True,pull=0.5,tie_harmonic_var=False)
	# 	print('LL: '+str(LL[-1]))
	# 	#print(gmm.weights)
	# 	print(gmm.means)
	# 	print(gmm.sigmas)
	# 	rec_spec[:,i]=get_vector(gmm,a,spec.shape[0])


	# plt.imshow(spec,origin='lower')
	# plt.savefig('/Users/Giaco/Desktop/encoder_images/30_o1'+str(j)+'.png', format='png', dpi=1000)
	# #plt.show()
	# plt.imshow(rec_spec,origin='lower')
	# plt.savefig('/Users/Giaco/Desktop/encoder_images/30_r1'+str(j)+'.png', format='png', dpi=1000)
	# #plt.show()


# vec_orig=transform(STFT)[:,135]
# samples,a=get_samples(vec_orig)
# gmm.fit(samples,art_init=True,pull=0)
print('------------')
vec_orig=transform(STFT)[:,150]
samples,a=get_samples(vec_orig)
vec_orig=samples_to_vector(samples,a)
LL=gmm.fit(samples,art_init=True,pull=0.5,tie_harmonic_var=False)
vector=get_vector(gmm,a,spec.shape[0])
print(gmm.weights)
print(gmm.means)
print(gmm.sigmas)
plt.plot(vector,'r-')
plt.plot(vec_orig,'b-')
plt.plot(gmm.means,np.zeros(len(gmm.means)),'b.',markersize=1.5,color='red')
plt.show()

# #-----workaround
# zhat=np.load('/Users/Giaco/Documents/Elektrotechnik-Master/Master Thesis/data_masterthesis/b7r16/b7r16_train/day08_b7r16/z_sequences/84052.npy')

# zhat=np.resize(zhat,(N,16))
# zhat[:,:]=z_sample
# #-----
# reconstructed_samples, reconstructed_audio = decode(zhat=zhat, netG=netG)
# plt.imshow(reconstructed_samples, origin='lower')
# plt.show()




