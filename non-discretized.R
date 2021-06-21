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
    return(runif(1, state, 1))
  else
    return(0)
}




# estimate the approximate stationary mean-field
stationary_mf <- function(policy)
{
  mean_field = NULL
  for(j in 1:mean_field_update)
  {
    #sampling from the initial state distribution (uniformly)
    state <- runif(num_players, 0, 1)    
    
    new_state <- NULL
    
    #get the mean field of the initial sampled states
    z <- mean(state)
    
    for (t in 0:trajectory_length) 
    {
      #get actions
      action <- 1*(state > policy)
      
      #states transition according to the given policy
      new_state <- mapply(next_state, state, action)
      
      #calculate the mean field at tth iteration
      z <- mean(state)
      
      state <- new_state
    }
    #get the average mean field
    mean_field <- c(mean_field, mean(state))
    
  }
  return (mean(mean_field))
}

# given the mean-field, compute the reward (cost) for a state-action pair
get_rewards <- function(state,action,mean_field)
{
    return(-(.2+mean_field) * state - .5 * action)
}



generate_episodes <- function(policy, mean_field, initial_state = NULL,
                              discount_factor=.9,trajectory_length=200)
{
  state <- ifelse(is.null(initial_state), runif(1,0,1), initial_state)
  cost <- 0
  for(episode in 1:trajectory_length)
  {
    action <- 1*(state > policy)
    new_state <- next_state(state, action)
    cost <- cost + discount_factor ^ episode * get_rewards(state, action, mean_field)
    state <- new_state
  }
  return(cost)
}



monte_carlo <- function(policy,mean_field, initial_state = NULL, 
                        num_iteration=200, discount_factor=.9)
{
  cost = NULL
  for(iteration in 1:num_iteration)
  {
    cost <- c(cost, generate_episodes(policy, mean_field, initial_state))
  }
  return(mean(cost))
}



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
    
    
    J_plus <- monte_carlo(theta_plus,z)
    J_minus <- monte_carlo(theta_minus,z)
    G <- eta / (2*beta) *(J_plus - J_minus)
    theta <- theta + alpha * G
    
    theta <- ifelse(theta>1,1,ifelse(theta<0,0,theta))
    iteration_history <- c(iteration_history,theta)
    print(c(episode,iteration_history[episode]))
  }
  
  print(c(iteration,theta))
  result <- rbind(result,iteration_history)
}
