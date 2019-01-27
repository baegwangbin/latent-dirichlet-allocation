% Answer to Question A

clear all
clc
load kos_doc_data.mat

W = max([A(:,2)]);  % number of unique words
D = max(A(:,1));    % number of documents in A

total_count = sum(A(:,3));
beta = zeros(1,W);

% count the number of appearance for each word index i

for I = 1:W 
     count_i = sum(A(A(:,2)==I,3));
     beta(I) = count_i/total_count;
end

[B,I] = sort(beta,'descend');
B = flip(B(1:20));
I = flip(I(1:20));
names = V(I);

%axis([0 1 0 20])

barh(B, 'k')
set(gca, 'YTickLabel', names, 'YTick', 1:20,'FontSize',12)

grid on;
title('20 Largest Probability Words', 'FontSize', 20, 'FontWeight', 'bold')
xlabel('Probability', 'FontSize', 15, 'FontWeight', 'bold');
ylabel('Words', 'FontSize', 15, 'FontWeight', 'bold');
