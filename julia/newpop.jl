## Cria população inicial
function newpop(Nind,CromLim)
let
    #fnp = Array{Float64}(undef,Nind,length(CromLim[:,1]))
    fnp = zeros(Nind,length(CromLim[:,1]))
    for i = 1:Nind
        for j=1:length(CromLim[:,1])
            inf=CromLim[j,1];
            sup=CromLim[j,2];
            fnp[i,j]=rand()*(sup-inf)+inf;
        end
    end
    return fnp
end
end
