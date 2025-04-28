%% Cria população inicial

function pop = newpop(Nind,CromLim)
    Ncrom = length(CromLim(:,1));
    pop = zeros(Nind,Ncrom);
    for i = 1:Nind
        for j=1:Ncrom
            Li=CromLim(j,1); Ls=CromLim(j,2);
            pop(i,j)=rand*(Ls-Li)+Li;
        end
    end 
end

%%