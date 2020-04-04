close all
clear
load('trained_models/c3m2_removeLow_kappaPerSubj4.mat')
load('removeLow.mat')
indices = setdiff(1:994,removeSubjects); % Get list of subjects in set
[sortedKappa,I] = sort(kappaPerSubj,'descend'); % Sort subjects by kappa
indices = indices(I);
p = polyfit(1:length(sortedKappa),sortedKappa',3); % Fit polynomial to curve
fitted = polyval(p,1:length(sortedKappa));
concavity = sign(gradient(gradient(fitted))); % Find point where sign of concavity changes
plot(sortedKappa)
hold on
plot(fitted)
%plot(concavity)
inflectionPoints = find(diff(concavity) ~= 0);
plot([inflectionPoints inflectionPoints],[1 0],'k--')
ylim([0 1])
xlabel('Patient','FontSize',18)
ylabel('Kappa','FontSize',18)
title('Sorted Kappa Per Patient','FontSize',20)
legend('Raw Kappa','Fitted 3rd Order Polynomial','Inflection Point')