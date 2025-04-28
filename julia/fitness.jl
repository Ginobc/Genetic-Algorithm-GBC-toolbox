## Fitness Function
function fitness(pop,k,qobj,N_ind)
let
	fity = zeros(N_ind)
	file = "vocal_tract_tmp";
	for i = 1:N_ind
		#println("N_ind = ", i)
		Ac = pop[i,:]';
		qdom = vocal_tract(Ac,file,k);
		# fity[i] = vecnorm(qdom-qobj); # julia v0.6 
		#fity[i] = 100. /(relative_error*norm(qdom-qobj)); # julia v0.7>
		fity[i] = 80. /norm(qdom-qobj); # julia v0.7>
	end
	return fity
end
end
