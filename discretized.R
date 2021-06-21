library(stats)

num_players <- 100
discretization_step <- .01
mean_field_update <- 100
trajectory_length <- 100
num_trajectory <- 200
num_runs <- 10
num_episode <- 60


next_state <- function(state,action)
{
  if(action == 0)
    return(sample(state_space[-(1:which(state_space == state))],1))
  else
    return(0)
}
state_space <- seq(from=0,to=1,by=discretization_step)

state_distribution <- rep(1/length(state_space),length(state_space))




# estimate the approximate stationary mean-field
stationary_mf <- function(policy)
{
  sum <- rep(0,length(state_space))
  for(j in 1:mean_field_update)
  {
    #sampling from the initial state distribution (uniformly)
    state <- sample(x=state_space,size = num_players, replace = TRUE, prob = state_distribution)    

    new_state <- NULL
    
    #get the mean field of the initial sampled states
    z <- state_distribution
    
    for (t in 0:trajectory_length) 
    {
      #get actions
      action <- 1*(state>=policy)

      #states transition according to the given policy
      new_state <- mapply(next_state, state, action)
      
      #calculate the mean field at tth iteration
      for(x in 1:length(state_space))
        z[x] <- (length(which(state<=state_space[x]))-length(which(state<=state_space[x-1])))/num_players
      
      state <- new_state
    }
    #get the average mean field
    sum <- sum + z

  }
  return (sum/mean_field_update)
}

# given the mean-field, compute the reward (cost) for a state-action pair
get_rewards <- function(state,action,mean_field)
{
  
  mean <- sum(state_space*mean_field, na.rm = TRUE)
  return(-(.2+mean) * state - .5 * action)
}



generate_episodes <- function(policy,state_distribution,mean_field, 
                              discount_factor=.9,trajectory_length=200)
{
  #state <- runif(1)
  state <- sample(state_space,size = 1,prob = state_distribution)
  states_visited <- NULL
  rewards_earned <- NULL
  count <- 1
  while (count<=trajectory_length) 
  {
    states_visited <- c(states_visited,ifelse(state==0,1,findInterval(state,state_space,left.open = TRUE)+1))
    action <- 1*(state >= policy)
    
    rewards_earned <- c(rewards_earned,get_rewards(state,action,mean_field))
    state <- next_state(state,action)
    count<-count+1
  }
  
  
  value <- rep(0,length(state_space))
  
  accessed_states <- rep(1,length(state_space))
  for (i in 1:length(state_space)) 
  {
    index <- match(i,states_visited)
    sum <- 0
    if(is.na(index))
    {
      accessed_states[i] <- 0
      next
    }
    
    for (j in index:trajectory_length) 
    {
      sum <- sum + discount_factor ^ (j-index) * rewards_earned[j]
    }
    value[i] <- sum
  }
  
  return(list('states_visited'=accessed_states,'rewards_earned'=value,'transition_history'=states_visited))
}



monte_carlo <- function(policy,state_distribution,mean_field, num_iteration=200, discount_factor=.9)
{
  num_visited <- rep(0,length(state_space))
  returns <- rep(0,length(state_space))
  
  for (i in 1:num_iteration) 
  {
    episode <- generate_episodes(policy,state_distribution,mean_field,discount_factor)
    num_visited <- num_visited + episode$states_visited
    returns <- returns + episode$rewards_earned
  }
  return(returns/num_visited)
}

sum(state_space * stationary_mf(.485162))
mean(monte_carlo(.485162, state_distribution, stationary_mf(.485162)))

result <- NULL

for (iteration in 1:num_runs) 
{
  #initialize theta randomly 
  theta <- runif(1)
  
  #record the estimate theta in this sample run
  iteration_history <- theta
  for(episode in 1:num_episode)
  {
    z <- stationary_mf(theta)
    eta <- sample(c(-1,1),1)
    
    alpha <- ifelse(episode<=50,1/(20+episode)^.602,1/(20+episode))
    beta <- ifelse(episode<=50,0.2/episode^.101,0.2/episode^.167)
    
    theta_plus <- ifelse(theta+eta*beta>1,1,ifelse(theta+eta*beta<0,0.01,theta+eta*beta))
    theta_minus <- ifelse(theta-eta*beta>1,1,ifelse(theta-eta*beta<0,0.01,theta-eta*beta))
    

    J_plus <- sum(monte_carlo(theta_plus,state_distribution,z)*state_distribution,na.rm = TRUE)
    J_minus <- sum(monte_carlo(theta_minus,state_distribution,z)*state_distribution, na.rm = TRUE)
    G <- eta / (2*beta) *(J_plus - J_minus)
    theta <- theta + alpha * G
    
    theta <- ifelse(theta>1,1,ifelse(theta<0,0,theta))
    iteration_history <- c(iteration_history,theta)
    print(c(episode,iteration_history[episode]))
  }
  
  print(c(iteration,theta))
  result <- rbind(result,iteration_history)
}
