import numpy as np
from GWSamplegen.waveform_utils import chirp_mass

#TODO: move to GWSamplegen.waveform_utils
def effective_spin(m1, m2, s1z, s2z):
	return (s1z * m1 + s2z * m2) / (m1 + m2)

#for non-gstlal pipelines, their bayes factor is: 
def get_f_over_b(far, snr_choice, far_star, snr_star):

	fground = 3 * snr_star**3 / (snr_choice**4)
	bground = far / far_star
	return fground / bground


#make a p-astro class to make saving/loading easier
class PastroCalculator:
	def __init__(self, template_bins, weights, W1_W0, counts, bin_type = "cm_q", far_star = 1e-4, snr_star = 8.5):
		#self.templates = templates
		self.weights = weights
		self.W1_W0 = W1_W0
		self.counts = counts
		self.bin_type = bin_type
		self.far_star = far_star
		self.snr_star = snr_star
		self.template_bins = template_bins

	def get_cm_bin_for_template_params(self, m1, m2):
		cm = chirp_mass(m1, m2)
		template_bin_idx = np.searchsorted([b[0][0] for b in self.template_bins], cm)
		return min(template_bin_idx, len(self.template_bins)-1)

	def get_cm_chi_bin_for_template_params(self, m1, m2, s1z, s2z):
		cm = chirp_mass(m1, m2)
		chi = effective_spin(m1, m2, s1z, s2z)
		for i in range(len(self.template_bins)):
			if cm >= self.template_bins[i][:,0].min() and cm <= self.template_bins[i][:,0].max():
				if chi >= (self.template_bins[i][:,3]*self.template_bins[i][:,1] + self.template_bins[i][:,4]*self.template_bins[i][:,2])/(self.template_bins[i][:,1] + self.template_bins[i][:,2]).min() and \
					chi <= (self.template_bins[i][:,3]*self.template_bins[i][:,1] + self.template_bins[i][:,4]*self.template_bins[i][:,2])/(self.template_bins[i][:,1] + self.template_bins[i][:,2]).max():
					return i
		return -1

	def get_cm_q_bin_for_template_params(self, m1, m2):
		cm = chirp_mass(m1, m2)
		q = m2/m1
		for i in range(len(self.template_bins)):
			if cm >= self.template_bins[i][:,0].min() and cm <= self.template_bins[i][:,0].max():
				if q >= (self.template_bins[i][:,2]/self.template_bins[i][:,1]).min() and \
					q <= (self.template_bins[i][:,2]/self.template_bins[i][:,1]).max():
					return i
		return -1
	
	def get_f_over_b(far, snr_choice, far_star, snr_star):

		fground = 3 * snr_star**3 / (snr_choice**4)
		bground = far / far_star
		return fground / bground
	
	def compute_p_astro(self, m1, m2, snr, far, s1z = 0, s2z = 0, verbose = False):
		"""
		Compute the astrophysical probability (P_astro) for a given event based on its parameters.
		
		"""
		#rewrite the compute_p_astro function to use the class attributes
		if self.bin_type == "cm":
			template_bin_idx = self.get_cm_bin_for_template_params(m1, m2)
		elif self.bin_type == "cm_chi_eff":
			template_bin_idx = self.get_cm_chi_bin_for_template_params(m1, m2, s1z, s2z)
		elif self.bin_type == "cm_q":
			template_bin_idx = self.get_cm_q_bin_for_template_params(m1, m2)
		
		if far > self.far_star:
			#print("FAR above threshold, setting P_astro to zero")
			return {"BNS": 0.0, "NSBH": 0.0, "BBH": 0.0, "Terr": 1.0}
		
		template_weights = self.W1_W0[template_bin_idx]
		#turn template_weights into a dict
		template_weights = {"BNS": template_weights[0], "NSBH": template_weights[1], "BBH": template_weights[2]}
		#TODO: confirm background weight is correct at 1/3.
		#background_weight = self.weights['Background'][template_bin_idx]

		#now compute f_over_b
		f_over_b = get_f_over_b(far, snr, self.far_star, self.snr_star)
		#print("f_over_b: ", f_over_b)

		#background_weight = 1/len(["BNS", "NSBH", "BBH"]) #to show where BG weight comes from
		#finally, we can compute P_astro
		
		numerator = { c: (f_over_b * self.counts[c] / self.counts['Terr']) * template_weights[c] for c in ["BNS", "NSBH", "BBH"] }
		denominator = (1+ f_over_b * np.sum( [ template_weights[c] * self.counts[c] / self.counts['Terr'] for c in ["BNS", "NSBH", "BBH"] ] ))

		p_astro = { c: numerator[c] / denominator for c in ["BNS", "NSBH", "BBH"] }

		p_terr = max(1- sum(p_astro.values()) , 0)
		p_astro['Terr'] = p_terr
		#print("P_astro : ", p_astro)
		return p_astro

		#return compute_p_astro(m1, m2, snr, far, self.templates, self.W1_W0, self.counts, s1z, s2z, verbose)

#test class