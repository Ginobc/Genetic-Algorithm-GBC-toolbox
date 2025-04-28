include("fitness.jl")
include("evolution_strategies.jl")
include("newpop.jl")

using JLD, Statistics, LinearAlgebra, Printf, Plots
pyplot()

## Inputs

# Do you want to continue the last optimization?
yesorno = 1;	# yes = 1; no = 0

## Optimization
# Optimization Parameters
N_ger = 10000;  # generation size
N_ind = 100;   	# chromosome size
N_diz = 50;    	# decimation step
p_diz = 0.15;  	# percentual of decimation
p_elit = 0.04;  # elitism probability
p_m = 0.1;    	# mutation probability
p_c = 0.65;     # crossover probability

## Fitness boundarys
CromLim = ones(15,2);

# Group 1
CromLim[1:4,1] .= 1;		# lower bound
CromLim[1:4,2] .= 3;		# upper bound

# Group 2
CromLim[5:9,1] .= 0.5;		# lower bound
CromLim[5:9,2] .= 3;		# upper bound

# Group 3
CromLim[10:15,1] .= 4;		# lower bound
CromLim[10:15,2] .= 16;		# upper bound

print(CromLim)

## pre allocated memory
if yesorno == 1
	ic = load("./myfile.jld", "i");
	erro = load("./myfile.jld", "erro");
	relative_error = load("./myfile.jld", "relative_error");
	pop = load("./myfile.jld", "pop");
	melhor = load("./myfile.jld", "melhor");
	medio = load("./myfile.jld", "medio");
	fittest = load("./myfile.jld", "fittest");
	fit = load("./myfile.jld", "fit");
else
	ic = 1;
	erro = 0;
	relative_error = 500;
	pop = newpop(N_ind,CromLim);
	melhor = zeros(N_ger);
	medio = zeros(N_ger);
	fittest = zeros(size(pop[1,:]));
	fit = zeros(N_ind);
end

Base.show(io::IO, f::Float64) = @printf(io, "%1.2f", f)		# Print two decimal digits
## GA Optimization
for i=ic:N_ger
	println("\n N_ger = ",i)
	save("./myfile.jld","melhor",melhor,"medio",medio,"fittest",fittest,"fit",fit,"i",i,"pop",pop,"relative_error",relative_error,"erro",erro)

	global ic
	global erro
	global relative_error
	global pop
	global melhor
	global medio
	global fittest
	global fit
	
	# chromosome fitness
	fit = fitness(pop, k, qdom, N_ind);
	
	# evolutionary strategies
	pop = evolution_strategies_old(pop,fit,p_elit,p_m,p_c,CromLim, N_ind);
	
	# best chromosome and its position
	best = findmax(fit);
	melhor[i] = best[1];
	medio[i] = mean(fit);
	fittest = pop[best[2],:];
	
	erro = 100*(fittest'-Ac)./ Ac;
	relative_error = mean(erro);
	println("\n - Aobj = ",Ac)
	println("\n - Afit = ",fittest')
	println("\n - melhor fitness = ",melhor[i-10:i])
	println("\n - error (%) = ",erro)
	println("\n - mean(error) (%) = ",relative_error)
	println("\n - std(error) (%) = ",std(erro))
	
	## decimattion
	if mod(i,N_diz)==0
		pop[round(Int64,N_ind*(1-p_diz))+1:N_ind,:] = newpop(N_ind-round(Int64,N_ind*(1-p_diz)),CromLim);
	end
end

ic = load("./myfile.jld", "i");
erro = load("./myfile.jld", "erro");
relative_error = load("./myfile.jld", "relative_error");
pop = load("./myfile.jld", "pop");
melhor = load("./myfile.jld", "melhor");
medio = load("./myfile.jld", "medio");
fittest = load("./myfile.jld", "fittest");
fit = load("./myfile.jld", "fit");
melhori = melhor[1:ic-1];
medioi = medio[1:ic-1];
plot(melhori,lab="fittest")
plot!(medioi,lab="mean",xlim=(0,ic-1))
xaxis!("Generation")
yaxis!("Fitness",:log10)

println("\n - Afit = ",fittest')
println("\n - melhor fitness = ",melhor[ic-10:ic-1])
println("\n - error (%) = ",erro)
println("\n - mean(error) (%) = ",relative_error)
println("\n - std(error) (%) = ",std(erro))