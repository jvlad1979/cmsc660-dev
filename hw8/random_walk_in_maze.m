function random_walk_in_maze()
close all
%% load data
data_down = readmatrix("maze_data_down.csv");
data_right = readmatrix("maze_data_right.csv");
data_down(1,:) = [];
data_down(:,1) = [];
data_right(1,:) = [];
data_right(:,1) = [];
%% draw maze
fig = 1;
draw_maze(data_down,data_right,fig)
% each maze cell is a node
% generate adjacency matrix
[m,n] = size(data_down);
N = m*n;
A = make_adjacency_matrix(data_down,data_right);
%% convert adjacency matrix to stochastic matrix 
row_sums = sum(A,2); % find rowsums of A
R = spdiags(1./row_sums,0,N,N);
P = R*A;
%% setup linear system for the committor problem
exitA = 1;
exitB = N;
x = zeros(N,1);
x(exitB) = 1;
I = speye(N);
L = P - I;
b = -L*x;
ind_exits = union(exitA,exitB);
ind_unknown = setdiff((1:N),ind_exits);
%% make the system symmetric positive definite
% L = R^{-1}A - I
% R^{1/2}LR^{-1/2} = R^{1/2}AR^{1/2} - I is symmetric
% Lsymm = R^{1/2}LR^{-1/2}, bsymm = R^{1/2}*b
% D = R^{1/2}
r_sqrt = sqrt(row_sums);
D = spdiags(r_sqrt,0,N,N);
Dinv = spdiags(1./r_sqrt,0,N,N);
Lsymm = D*L*Dinv;
bsymm = D*b;
%% solve the linear system L*x(ind_unknown) = b
x(ind_unknown) = Lsymm(ind_unknown,ind_unknown)\bsymm(ind_unknown);
solution = x(ind_unknown);
x(ind_unknown) = Dinv(ind_unknown,ind_unknown)*x(ind_unknown);
%% visualize the solution
figure(3);
committor = reshape(x,[m,n]);
imagesc(committor);
draw_maze(data_down,data_right,3)
%% find eigenvalues of Lsymm
evals = sort(real(eig(full(-(Lsymm)))),'descend');
figure(4);
plot(evals,'.','Markersize',15);
grid on
set(gca,'fontsize',20)
xlabel('eigenvalue index','FontSize',20);
ylabel('eigenvalue','FontSize',20);
%% save the solution
A = -Lsymm(ind_unknown,ind_unknown);
b = -bsymm(ind_unknown);
save("maze_linear_system.mat","A","b","solution");

end

%%
function draw_maze(data_down,data_right,fig)
[m,n] = size(data_down);
figure(fig);
hold on;
line_width = 3;
col = 'k';
% plot outer lines
plot(0.5+(1:n),0.5+zeros(1,n),'color',col,'Linewidth',line_width);
plot(0.5+(0:n-1),0.5+m*ones(1,n),'color',col,'Linewidth',line_width);
plot(0.5+zeros(1,m),0.5+(1:m),'color',col,'Linewidth',line_width);
plot(0.5+m*ones(1,n),0.5+(0:m-1),'color',col,'Linewidth',line_width);
% plot vertical lines
for i = 1 : m
    for j = 1 : n-1
        if data_right(i,j) == 0
            plot(0.5+[j,j],0.5+[i-1,i],'color',col,'Linewidth',line_width);
        end
    end
end
% plot horizontal lines
for j = 1 : n
    for i = 1 : m-1
        if data_down(i,j) == 0
            plot(0.5+[j-1,j],0.5+[i,i],'color',col,'Linewidth',line_width);
        end
    end
end
axis ij
axis off
daspect([1,1,1])
end
%%
function A = make_adjacency_matrix(data_down,data_right)
[m,n] = size(data_down);
mn = m*n;
A = sparse(mn);
for i = 1 : m
    for j = 1 : n-1
        if data_right(i,j) == 1
            ind = (j-1)*m + i;
            A(ind,ind+m) = 1;
            A(ind+m,ind) = 1;
        end
    end
end
for j = 1 : n
    for i = 1 : m-1
        if data_down(i,j) == 1
            ind = (j-1)*m + i;
            A(ind,ind+1) = 1;
            A(ind+1,ind) = 1;
        end
    end
end
end



