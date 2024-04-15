delta(-10..10).
player(0).

#constant(threshold, -10..10).

#modeha(left).



#modeb(1,target_x(var(delta)), (positive)).
#modeb(3,var(delta) < const(threshold)).

#modeb(1,player_x(var(delta)), (positive)).
%#modeb(var(delta) < const(threshold)).

#modeb(1,comm(var(delta)), (positive)).
%#modeb(var(delta) < const(threshold)).

#maxv(3).
#maxhl(1).

