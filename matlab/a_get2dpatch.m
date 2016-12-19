function [neighborInd] = a_get2dpatch(imdb,coord,k)
 siz = size(imdb);                              %# matrix size
%# 3D point location

%# neighboring points
%# radius size
k=(k-1)/2;
[sx,sy] = ndgrid(-k:k,-k:k);          %# steps to get to neighbors
xy = bsxfun(@plus, coord, [sx(:) sy(:)]);  %# add shift
xy = bsxfun(@min, max(xy,1), siz);          %# clamp coordinates within range
xy = unique(xy,'rows');                     %# remove duplicates
xy(ismember(xy,coord,'rows'),:) = [];           %# remove point itself

%# show solution
% figure
% line(p(1), p(2), p(3), 'Color','r', ...
%     'LineStyle','none', 'Marker','.', 'MarkerSize',50)
% line(xyz(:,1), xyz(:,2), xyz(:,3), 'Color','b', ...
%     'LineStyle','none', 'Marker','.', 'MarkerSize',20)
% view(3), grid on, box on, axis equal
% axis([1 siz(1) 1 siz(2) 1 siz(3)])
% xlabel x, ylabel y, zlabel z

neighborInd = sub2ind(siz, xy(:,1), xy(:,2));
% if numel(linearInd)==(dim*dim)
% n=reshape(imdb(linearInd),[dim dim dim]);
% neighbors=squeeze(n(k,:,:));
% ind=index;
% else
%     neighbors=1;
%     ind=0;
% end