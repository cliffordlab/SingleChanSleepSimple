function splitAtInflection(kappas,omit,totalSubj,removeHigh,removeLow,validationSubjects)
% Load a list of kappa values for each patient along from file 'kappas'
% along with a list of patients not to include in analysis 'omit'. The data 
% will then be sorted from highest to lowest and split at the inflection point. 
% New files will be generated containing original list of patients to remove
% along with all of those above and below the inflection point, which will
% be named removeHigh and removeLow, respectively.
% Load data
load(kappas,'kappaPerSubj');
load(omit,'removeSubjects');
load(validationSubjects,'validationSubjects');

indices = setdiff(1:str2num(totalSubj),union(removeSubjects,validationSubjects)); % Get list of subjects in set
[sortedKappa,I] = sort(kappaPerSubj,'descend'); % Sort subjects by kappa
indices = indices(I);
p = polyfit(1:length(sortedKappa),sortedKappa',3); % Fit polynomial to curve
fitted = polyval(p,1:length(sortedKappa));
concavity = sign(gradient(gradient(fitted))); % Find point where sign of concavity changes
inflectionPoints = find(diff(concavity) ~= 0); % Calculate inflection points

% % Plot
% plot(sortedKappa)
% hold on
% plot(fitted)
% plot([inflectionPoints inflectionPoints],[1 0],'k--')
% ylim([0 1])
% xlabel('Patient','FontSize',20)
% ylabel('Kappa','FontSize',20)
% title('Sorted Kappa Per Patient','FontSize',22)
% lgd = legend('Raw Kappa','Fitted 3rd Order Polynomial','Inflection Point');
% lgd.FontSize = 20;
% a = get(gca,'XTickLabel');
% b = get(gca,'YTickLabel');
% set(gca,'XTickLabel',a,'fontsize',16)
% set(gca,'YTickLabel',b,'fontsize',16)

disp(['Number of inflection points = ' num2str(length(inflectionPoints))])

% Add list of patients above or below inflection point to lists of patients
% to remove on future iterations
originalRemoved = removeSubjects; 
removeSubjects = [originalRemoved indices(1:inflectionPoints(1))];
save(removeHigh,'removeSubjects','-v7.3')

removeSubjects = [originalRemoved indices((inflectionPoints(1)+1):end)];
save(removeLow,'removeSubjects','-v7.3')
end