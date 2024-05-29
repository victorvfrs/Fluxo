def save_model(epoch, generator, discriminator, optimizer_generator, optimizer_discriminator, file="gan_checkpoint.pth.tar"):
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_generator_state_dict': optimizer_generator.state_dict(),
        'optimizer_discriminator_state_dict': optimizer_discriminator.state_dict(),
    }, file)

def load_model(file, generator, discriminator, optimizer_generator, optimizer_discriminator):
    checkpoint = torch.load(file)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_generator.load_state_dict(checkpoint['optimizer_generator_state_dict'])
    optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator_state_dict'])
    return checkpoint['epoch']