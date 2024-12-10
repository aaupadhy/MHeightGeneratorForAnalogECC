# Generic mutation function
def mutate(tensor, mutation_range=(-1, 1),flag=0):
    if flag == 0:
      return tensor + torch.randint(mutation_range[0], mutation_range[1] + 1, tensor.shape, device=tensor.device)
    else:
      return tensor + torch.randn(tensor.shape, device=tensor.device)

# Generic crossover function
def crossover(tensorA, tensorB, perturbation_range=(-1, 1)):
    
    mask = torch.randint(0, 2, tensorA.shape, device=tensorA.device, dtype=torch.bool)
    result = torch.where(mask, tensorA, tensorB)
    perturbation = torch.randint(perturbation_range[0], perturbation_range[1] + 1, tensorA.shape, device=tensorA.device)
    return result + perturbation