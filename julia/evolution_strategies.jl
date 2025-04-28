## Estratégias Evolutivas
function evolution_strategies(pop,fit,p_elit,p_m,p_c,CromLim,N_ind)
	let
		j = 0;
		
		N_elit = round(Int64,N_ind*p_elit)
		if N_elit < 1
			N_elit = 1;
		end
		
		new_pop = zeros(size(pop))
		q = zeros(N_ind)
		seleciona = zeros(Int64,N_ind)
		
		# Selection (Roullete-Wheel):
		P = fit/sum(fit);
		r = rand(N_ind);
	
		for i = 1:N_ind
			q[i] = sum(P[1:i]);     # Probabilidades Acumulativas
			if r[i] < q[i]
				j += 1;
				seleciona[j] = i;
			end
		end
		
		# Elitism
		pos = sortperm(fit,rev=true);
	
		ii = 1;
		jj = 1;
		#while ii <= N_elit			# (18/01/2019) Implementação de técnica para não repetir indivíduos de elite
		#	if ii == 1
		#		new_pop[ii,:] = pop[pos[jj],:];
		#		ii += 1;
		#	else
		#		if fit[jj]==fit[jj-1]
		#		
		#		else
		#			new_pop[ii,:] = pop[pos[jj],:];
		#			ii += 1;
		#		end
		#	end
		#	jj += 1;
		#end
		
		for i = 1:N_elit
		   new_pop[ii,:] = pop[pos[ii],:];
		   ii += 1;
		end

		# Crossover and Mutation
		while ii <= N_ind
			if p_m <= rand()
				pos_pai = seleciona[rand(1:j)];
				pos_mae = seleciona[rand(1:j)];
				pai = pop[pos_pai,:];
				mae = pop[pos_mae,:];
				if p_c >= rand()
					alpha = 0.25;
					a2 = -alpha; b2 = 1+alpha;
					beta = a2 + (b2-a2).*rand();
					new_pop[ii,:] = pai+beta*(mae-pai);
					# Resolução do problema de extrapolação dos limites
					for k=1:size(CromLim,1)
						if new_pop[ii,k]<CromLim[k,1]
							new_pop[ii,k]=CromLim[k,1];
						end
						if new_pop[ii,k]>CromLim[k,2]
							new_pop[ii,k]=CromLim[k,2];
						end
					end
				else
					if fit[pos_pai]>fit[pos_mae]
						new_pop[ii,:] = pai;
					else
						new_pop[ii,:] = mae;
					end
				end
			else
				new_pop[ii,:] = newpop(1,CromLim);
			end
			ii += 1;
		end
		return new_pop
	end
end
